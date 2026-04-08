from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class StepResponseData:
    t: np.ndarray
    y: np.ndarray
    u: np.ndarray | None = None
    ambient: np.ndarray | None = None
    step_index: int | None = None
    step_amplitude: float = 1.0


@dataclass
class SecondOrderStepModel:
    gain: float
    zeta: float
    omega_n: float
    delay: float
    overshoot: float
    rise_time: float
    settling_time: float
    peak_time: float
    y0: float
    y_ss: float
    step_amplitude: float

    @property
    def denominator(self) -> np.ndarray:
        return np.array([1.0, 2.0 * self.zeta * self.omega_n, self.omega_n**2])

    @property
    def numerator(self) -> np.ndarray:
        return np.array([self.gain * self.omega_n**2])

    @property
    def polynomial(self) -> str:
        return f"s^2 + {2.0 * self.zeta * self.omega_n:.6g}s + {self.omega_n**2:.6g}"

    @property
    def transfer_function(self) -> str:
        return (
            f"G(s) = ({self.gain:.6g} * {self.omega_n**2:.6g}) / "
            f"(s^2 + {2.0 * self.zeta * self.omega_n:.6g}s + {self.omega_n**2:.6g})"
        )


def load_step_response_csv(
    csv_path: str | Path,
    time_column: str = "t",
    output_column: str = "y",
) -> StepResponseData:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")

    rows = np.genfromtxt(path, delimiter=",", names=True, dtype=float)
    if rows.size == 0:
        raise ValueError(f"CSV has no data rows: {path}")

    column_names = rows.dtype.names
    if column_names is None:
        raise ValueError("CSV header not found. Expected at least columns t,y")
    if time_column not in column_names or output_column not in column_names:
        raise ValueError(
            f"CSV must contain '{time_column}' and '{output_column}' columns. Found: {column_names}"
        )

    t = np.atleast_1d(np.asarray(rows[time_column], dtype=float))
    y = np.atleast_1d(np.asarray(rows[output_column], dtype=float))
    if t.shape[0] != y.shape[0]:
        raise ValueError("Time and output columns must have the same length")
    if t.shape[0] < 5:
        raise ValueError("Need at least 5 samples to identify a second-order model")

    return StepResponseData(t=t, y=y)


def load_step_response_txt(
    txt_path: str | Path,
    time_col: int = 0,
    output_col: int = 1,
    control_col: int = 2,
    ambient_col: int = 3,
) -> StepResponseData:
    path = Path(txt_path)
    if not path.exists():
        raise FileNotFoundError(f"Input TXT not found: {path}")

    data = np.atleast_2d(np.loadtxt(path, dtype=float))
    if data.shape[1] <= max(time_col, output_col, control_col):
        raise ValueError("TXT must have at least time, output and control columns")

    t = np.asarray(data[:, time_col], dtype=float)
    y = np.asarray(data[:, output_col], dtype=float)
    u = np.asarray(data[:, control_col], dtype=float)
    ambient = data[:, ambient_col] if data.shape[1] > ambient_col else None

    if t.shape[0] < 5:
        raise ValueError("Need at least 5 samples to identify a second-order model")

    du = np.diff(u, prepend=u[0])
    change_idx = np.where(np.abs(du) > 1e-12)[0]
    step_index = int(change_idx[0]) if change_idx.size else None
    step_amplitude = float(du[step_index]) if step_index is not None else 1.0
    if np.isclose(step_amplitude, 0.0):
        step_amplitude = 1.0

    return StepResponseData(
        t=t,
        y=y,
        u=u,
        ambient=ambient,
        step_index=step_index,
        step_amplitude=step_amplitude,
    )


def load_step_response_data(path: str | Path) -> StepResponseData:
    ext = Path(path).suffix.lower()
    if ext == ".csv":
        return load_step_response_csv(path)
    if ext == ".txt":
        return load_step_response_txt(path)
    raise ValueError("Unsupported input format. Use .csv or .txt")


def _tail_average(values: np.ndarray, count: int) -> float:
    tail = values[-count:] if values.shape[0] >= count else values
    return float(np.mean(tail))


def _estimate_delay(t: np.ndarray, y: np.ndarray, y0: float, y_ss: float) -> float:
    delta = y_ss - y0
    if np.isclose(delta, 0.0):
        return 0.0

    slope = np.abs(np.diff(y, prepend=y[0]))
    slope_peak = float(np.max(slope)) if slope.size else 0.0
    if slope_peak > 0.0:
        hits = np.where(slope >= 0.05 * slope_peak)[0]
    else:
        hits = np.array([], dtype=int)

    if not hits.size:
        direction = 1.0 if delta >= 0.0 else -1.0
        threshold = y0 + direction * 0.02 * abs(delta)
        if direction > 0.0:
            hits = np.where(y >= threshold)[0]
        else:
            hits = np.where(y <= threshold)[0]
    return float(t[int(hits[0])]) if hits.size else float(t[0])


def _estimate_rise_time(
    t: np.ndarray, y: np.ndarray, y0: float, y_ss: float, delay: float
) -> float:
    delta = y_ss - y0
    if np.isclose(delta, 0.0):
        return 0.0

    direction = 1.0 if delta >= 0.0 else -1.0
    y10 = y0 + direction * 0.1 * abs(delta)
    y90 = y0 + direction * 0.9 * abs(delta)
    post_delay = t >= delay
    idx = np.where(post_delay & ((y >= y10) if direction > 0 else (y <= y10)))[0]
    if not idx.size:
        return 0.0
    t10 = float(t[int(idx[0])])
    idx = np.where(post_delay & ((y >= y90) if direction > 0 else (y <= y90)))[0]
    if not idx.size:
        return 0.0
    t90 = float(t[int(idx[0])])
    return max(0.0, t90 - t10)


def _estimate_settling_time(t: np.ndarray, y: np.ndarray, y_ss: float, band: float = 0.02) -> float:
    error = np.abs(y - y_ss)
    tol = band * max(1.0, abs(y_ss), float(np.max(error)) if error.size else 1.0)
    outside = np.where(error > tol)[0]
    if not outside.size:
        return float(t[0])
    last_outside = int(outside[-1])
    if last_outside + 1 >= t.shape[0]:
        return float(t[-1])
    return float(t[last_outside + 1])


def _estimate_zeta_from_overshoot(overshoot: float) -> float:
    if overshoot <= 0.0:
        return 1.0
    log_term = np.log(overshoot)
    return float(-log_term / np.sqrt(np.pi**2 + log_term**2))


def _estimate_omega_n(t_peak: float, zeta: float, settling_time: float) -> float:
    if 0.0 < zeta < 1.0 and t_peak > 0.0:
        return float(np.pi / (t_peak * np.sqrt(1.0 - zeta**2)))
    if settling_time > 0.0:
        return float(4.0 / max(1e-9, zeta * settling_time))
    return 1.0


def estimate_second_order_step_model(data: StepResponseData) -> SecondOrderStepModel:
    t = np.asarray(data.t, dtype=float)
    y = np.asarray(data.y, dtype=float)
    if t.shape[0] != y.shape[0]:
        raise ValueError("Time and output arrays must have the same length")

    head_count = max(3, min(10, t.shape[0] // 10 or 3))
    tail_count = max(3, min(10, t.shape[0] // 10 or 3))
    y0 = _tail_average(y[:head_count], head_count)
    y_ss = _tail_average(y, tail_count)

    step_amplitude = float(data.step_amplitude)
    if np.isclose(step_amplitude, 0.0):
        step_amplitude = 1.0

    if data.step_index is not None and 0 <= data.step_index < t.shape[0]:
        delay = float(t[data.step_index])
    else:
        delay = _estimate_delay(t, y, y0, y_ss)

    direction = 1.0 if (y_ss - y0) >= 0.0 else -1.0
    peak_idx = int(np.argmax(direction * (y - y0)))
    peak_time = float(t[peak_idx] - delay)
    peak_value = float(y[peak_idx])
    response_delta = y_ss - y0
    overshoot = 0.0
    if not np.isclose(response_delta, 0.0):
        overshoot = max(0.0, direction * (peak_value - y_ss) / abs(response_delta))

    rise_time = _estimate_rise_time(t, y, y0, y_ss, delay)
    settling_time = _estimate_settling_time(t, y, y_ss)
    zeta = _estimate_zeta_from_overshoot(overshoot)
    omega_n = _estimate_omega_n(max(1e-9, peak_time), zeta, settling_time)
    gain = float(response_delta / step_amplitude)

    return SecondOrderStepModel(
        gain=gain,
        zeta=zeta,
        omega_n=omega_n,
        delay=delay,
        overshoot=overshoot,
        rise_time=rise_time,
        settling_time=settling_time,
        peak_time=max(0.0, peak_time),
        y0=y0,
        y_ss=y_ss,
        step_amplitude=step_amplitude,
    )


def second_order_summary(model: SecondOrderStepModel) -> dict[str, float]:
    return {
        "gain": float(model.gain),
        "zeta": float(model.zeta),
        "omega_n": float(model.omega_n),
        "delay": float(model.delay),
        "overshoot": float(model.overshoot),
        "rise_time": float(model.rise_time),
        "settling_time": float(model.settling_time),
        "peak_time": float(model.peak_time),
    }
