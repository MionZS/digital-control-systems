from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class ImpulseResponseModel:
    """Discrete-time FIR model identified from impulse-response data."""

    dt: float
    numerator: np.ndarray
    denominator: np.ndarray

    @property
    def dc_gain(self) -> float:
        return float(np.sum(self.numerator))


def load_impulse_response_csv(
    csv_path: str | Path,
    time_column: str = "t",
    output_column: str = "y",
) -> tuple[np.ndarray, np.ndarray]:
    """Load impulse-response data from CSV with explicit time and output columns."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")

    rows = np.genfromtxt(path, delimiter=",", names=True, dtype=float)
    if rows.size == 0:
        raise ValueError(f"CSV has no data rows: {path}")

    column_names = rows.dtype.names
    if column_names is None:
        raise ValueError("CSV header not found. Expected columns: t,y")
    if time_column not in column_names or output_column not in column_names:
        raise ValueError(
            f"CSV must contain '{time_column}' and '{output_column}' columns. Found: {column_names}"
        )

    t = np.atleast_1d(np.asarray(rows[time_column], dtype=float))
    y = np.atleast_1d(np.asarray(rows[output_column], dtype=float))

    if t.shape[0] != y.shape[0]:
        raise ValueError("Time and output columns must have the same length")
    if t.shape[0] < 3:
        raise ValueError("Need at least 3 samples to identify an impulse model")

    return t, y


def load_impulse_response_txt(
    txt_path: str | Path,
    time_col: int = 0,
    internal_temp_col: int = 1,
    control_col: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Load whitespace-separated thermal data and estimate impulse response.

    Expected layout per row:
    time[s] internal_temp control_signal ambient_temp
    """
    path = Path(txt_path)
    if not path.exists():
        raise FileNotFoundError(f"Input TXT not found: {path}")

    data = np.loadtxt(path, dtype=float)
    data = np.atleast_2d(data)
    if data.shape[1] < 3:
        raise ValueError(
            "TXT must have at least 3 columns: time internal_temp control_signal [ambient_temp]"
        )

    t = np.asarray(data[:, time_col], dtype=float)
    y_internal = np.asarray(data[:, internal_temp_col], dtype=float)
    u = np.asarray(data[:, control_col], dtype=float)

    if t.shape[0] < 3:
        raise ValueError("Need at least 3 samples to identify an impulse model")

    du = np.diff(u, prepend=u[0])
    change_idx = np.where(np.abs(du) > 1e-12)[0]
    if change_idx.size == 0:
        raise ValueError("No control step detected in TXT input")

    step_idx = int(change_idx[0])
    delta_u = float(du[step_idx])
    if np.isclose(delta_u, 0.0):
        raise ValueError("Detected control step has zero amplitude")

    y0 = float(np.mean(y_internal[:step_idx])) if step_idx > 0 else float(y_internal[0])
    step_response = (y_internal - y0) / delta_u

    # Discrete-time derivative of the step response approximates impulse response.
    h = np.diff(step_response, prepend=0.0)
    if step_idx > 0:
        h[:step_idx] = 0.0

    return t, h


def load_impulse_response_data(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load impulse-response-compatible data from CSV or TXT."""
    ext = Path(path).suffix.lower()
    if ext == ".csv":
        return load_impulse_response_csv(path)
    if ext == ".txt":
        return load_impulse_response_txt(path)
    raise ValueError("Unsupported input format. Use .csv or .txt")


def infer_dt(t: np.ndarray) -> float:
    """Infer a representative sampling period from time stamps."""
    dt_candidates = np.diff(np.asarray(t, dtype=float))
    if np.any(dt_candidates <= 0.0):
        raise ValueError("Time vector must be strictly increasing")
    return float(np.median(dt_candidates))


def identify_fir_from_impulse(
    t: np.ndarray,
    y: np.ndarray,
    dt: float | None = None,
    max_taps: int | None = None,
) -> ImpulseResponseModel:
    """Identify a discrete FIR model where impulse response equals measured output."""
    inferred_dt = infer_dt(t)
    model_dt = inferred_dt if dt is None else float(dt)
    if model_dt <= 0.0:
        raise ValueError("dt must be positive")

    h = np.asarray(y, dtype=float)
    if max_taps is not None:
        if max_taps < 1:
            raise ValueError("max_taps must be >= 1")
        h = h[:max_taps]

    return ImpulseResponseModel(dt=model_dt, numerator=h, denominator=np.array([1.0]))


def impulse_summary(model: ImpulseResponseModel) -> dict[str, float]:
    """Compute scalar summary metrics used by the notebook UI."""
    peak_idx = int(np.argmax(np.abs(model.numerator)))
    return {
        "dt": float(model.dt),
        "num_taps": float(model.numerator.shape[0]),
        "dc_gain": float(model.dc_gain),
        "peak_amplitude": float(model.numerator[peak_idx]),
        "peak_index": float(peak_idx),
    }
