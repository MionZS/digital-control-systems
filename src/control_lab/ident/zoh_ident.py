from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import cont2discrete, lfilter

from control_lab.ident.second_order import StepResponseData, estimate_second_order_step_model


@dataclass
class SignalData:
    t: np.ndarray
    v: np.ndarray


@dataclass
class ZOHIdentificationResult:
    dt: float
    na: int
    nb: int
    nk: int
    den_z: np.ndarray
    num_z: np.ndarray
    den_s: np.ndarray
    num_s: np.ndarray
    fit_rmse: float
    y_hat: np.ndarray
    gain: float | None = None
    zeta: float | None = None
    omega_n: float | None = None
    delay: float | None = None
    method: str = "arx-matched"


def _parse_lvm_rows(file_path: Path) -> np.ndarray:
    numeric_rows: list[list[float]] = []
    start_parsing = False

    with file_path.open("r", encoding="utf-8", errors="ignore") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue

            if "***End_of_Header***" in line:
                start_parsing = True
                continue

            if not start_parsing and line.startswith("#"):
                continue

            # LVM commonly uses comma as decimal separator, so do not split on comma.
            tokens = re.split(r"[\t; ]+", line)
            values: list[float] = []
            for token in tokens:
                if not token:
                    continue
                try:
                    values.append(float(token.replace(",", ".")))
                except ValueError:
                    continue

            if len(values) >= 2:
                numeric_rows.append(values)

    if not numeric_rows:
        raise ValueError(f"No numeric rows found in LVM file: {file_path}")

    max_cols = max(len(r) for r in numeric_rows)
    arr = np.full((len(numeric_rows), max_cols), np.nan, dtype=float)
    for i, row in enumerate(numeric_rows):
        arr[i, : len(row)] = row
    return arr


def load_signal_file(path: str | Path) -> SignalData:
    """Load signal file with either 1-column (value) or 2+-columns (time, value)."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Signal file not found: {file_path}")

    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        arr = np.genfromtxt(file_path, delimiter=",", dtype=float)
    elif suffix == ".lvm":
        arr = _parse_lvm_rows(file_path)
    else:
        arr = np.loadtxt(file_path, dtype=float)

    arr = np.atleast_2d(arr)
    if arr.shape[1] >= 2:
        finite_mask = np.isfinite(arr[:, 0]) & np.isfinite(arr[:, 1])
    else:
        finite_mask = np.isfinite(arr[:, 0])
    arr = arr[finite_mask]

    if arr.shape[0] < 2:
        raise ValueError("Need at least two rows in signal file")

    if arr.shape[1] == 1:
        v = np.asarray(arr[:, 0], dtype=float)
        t = np.arange(v.shape[0], dtype=float)
    else:
        t = np.asarray(arr[:, 0], dtype=float)
        v = np.asarray(arr[:, 1], dtype=float)

    if t.shape[0] != v.shape[0]:
        raise ValueError("Time and value columns must have equal length")

    return SignalData(t=t, v=v)


def _detect_step(u: np.ndarray) -> tuple[int, float]:
    du = np.diff(u, prepend=u[0])
    idx = np.where(np.abs(du) > 1e-12)[0]
    if idx.size == 0:
        return 0, 1.0
    step_idx = int(idx[0])
    amp = float(du[step_idx])
    if np.isclose(amp, 0.0):
        amp = 1.0
    return step_idx, amp


def continuous_to_discrete_zoh(
    num_s: np.ndarray, den_s: np.ndarray, dt: float
) -> tuple[np.ndarray, np.ndarray]:
    """Exact ZOH discretization of G(s) -> G(z).

    This is consistent with the lecture procedure based on sampled step response:
    G(z) = (z-1)/z * Z{ L^-1( G(s)/s ) sampled at t = k*dt }.
    """
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    numd, dend, _ = cont2discrete((num_s, den_s), dt=dt, method="zoh")
    num_z = np.asarray(numd).reshape(-1)
    den_z = np.asarray(dend).reshape(-1)
    if not np.isclose(den_z[0], 1.0):
        num_z = num_z / den_z[0]
        den_z = den_z / den_z[0]
    return num_z, den_z


def identify_zoh_from_second_order(
    control_path: str | Path,
    output_path: str | Path,
) -> tuple[ZOHIdentificationResult, np.ndarray, np.ndarray, np.ndarray]:
    u_signal = load_signal_file(control_path)
    y_signal = load_signal_file(output_path)
    t, u, y = align_signals(u_signal, y_signal)
    dt = infer_dt(t)

    step_index, step_amplitude = _detect_step(u)
    data = StepResponseData(
        t=t,
        y=y,
        u=u,
        step_index=step_index,
        step_amplitude=step_amplitude,
    )
    model = estimate_second_order_step_model(data)
    y0_guess = float(np.mean(y[: max(2, min(20, step_index if step_index > 0 else 2))]))

    def _simulate_with_params(
        params: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        gain_p = float(params[0])
        zeta_p = float(params[1])
        omega_n_p = float(params[2])
        delay_p = float(params[3])
        y0_p = float(params[4])

        num_s_p = np.array([gain_p * omega_n_p**2], dtype=float)
        den_s_p = np.array([1.0, 2.0 * zeta_p * omega_n_p, omega_n_p**2], dtype=float)
        num_z_p, den_z_p = continuous_to_discrete_zoh(num_s=num_s_p, den_s=den_s_p, dt=dt)
        delay_samples_p = max(0, int(round(delay_p / dt)))
        if delay_samples_p > 0:
            num_z_p = np.concatenate([np.zeros(delay_samples_p, dtype=float), num_z_p])

        y_hat_p = lfilter(num_z_p, den_z_p, u) + y0_p
        return num_s_p, den_s_p, num_z_p, den_z_p, y_hat_p

    # Use characteristic-based estimate as initial guess, then refine by least squares.
    p0 = np.array(
        [
            max(1e-6, float(model.gain)),
            float(np.clip(model.zeta, 0.05, 2.5)),
            max(1e-4, float(model.omega_n)),
            max(0.0, float(model.delay)),
            y0_guess,
        ],
        dtype=float,
    )

    k_scale = max(
        1.0, abs(float(model.gain)), abs(float((y[-1] - y0_guess) / max(step_amplitude, 1e-9)))
    )
    delay_max = float(max(0.0, t[-1] - t[0]))

    lb = np.array([1e-8, 0.01, 1e-5, 0.0, float(np.min(y) - 2.0 * np.std(y))], dtype=float)
    ub = np.array(
        [50.0 * k_scale, 3.0, 5.0, delay_max, float(np.max(y) + 2.0 * np.std(y))], dtype=float
    )

    def _residuals(params: np.ndarray) -> np.ndarray:
        _, _, _, _, y_hat_p = _simulate_with_params(params)
        return y_hat_p - y

    opt = least_squares(_residuals, p0, bounds=(lb, ub), method="trf", loss="soft_l1")
    p_opt = opt.x if opt.success else p0
    num_s, den_s, num_z, den_z, y_hat = _simulate_with_params(p_opt)
    gain_opt, zeta_opt, omega_n_opt, delay_opt, _ = [float(v) for v in p_opt]

    rmse = float(np.sqrt(np.mean((y - y_hat) ** 2)))
    result = ZOHIdentificationResult(
        dt=dt,
        na=den_z.size - 1,
        nb=num_z.size,
        nk=0,
        den_z=den_z,
        num_z=num_z,
        den_s=den_s,
        num_s=num_s,
        fit_rmse=rmse,
        y_hat=y_hat,
        gain=gain_opt,
        zeta=zeta_opt,
        omega_n=omega_n_opt,
        delay=delay_opt,
        method="second-order-zoh-step-invariant",
    )
    return result, t, u, y


def infer_dt(t: np.ndarray) -> float:
    dt_values = np.diff(np.asarray(t, dtype=float))
    if dt_values.size == 0:
        raise ValueError("Need at least two samples to infer dt")
    if np.any(dt_values <= 0.0):
        raise ValueError("Time vector must be strictly increasing")
    return float(np.median(dt_values))


def align_signals(
    input_signal: SignalData, output_signal: SignalData
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dt_u = infer_dt(input_signal.t)
    dt_y = infer_dt(output_signal.t)
    dt = min(dt_u, dt_y)

    t0 = max(float(input_signal.t[0]), float(output_signal.t[0]))
    tf = min(float(input_signal.t[-1]), float(output_signal.t[-1]))
    if tf <= t0:
        raise ValueError("Input and output signals do not overlap in time")

    n = int(np.floor((tf - t0) / dt)) + 1
    t = t0 + np.arange(n, dtype=float) * dt

    u = np.interp(t, input_signal.t, input_signal.v)
    y = np.interp(t, output_signal.t, output_signal.v)
    return t, u, y


def estimate_arx(
    u: np.ndarray, y: np.ndarray, na: int, nb: int, nk: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if na < 1 or nb < 1:
        raise ValueError("na and nb must be >= 1")
    if nk < 0:
        raise ValueError("nk must be >= 0")
    if u.shape[0] != y.shape[0]:
        raise ValueError("Input and output lengths must match")

    n = y.shape[0]
    k0 = max(na, nk + nb - 1)
    if n <= k0 + 2:
        raise ValueError("Not enough samples for requested ARX orders")

    rows = []
    targets = []
    for k in range(k0, n):
        phi_y = [-y[k - i] for i in range(1, na + 1)]
        phi_u = [u[k - nk - j] for j in range(nb)]
        rows.append(phi_y + phi_u)
        targets.append(y[k])

    phi = np.asarray(rows, dtype=float)
    y_target = np.asarray(targets, dtype=float)
    theta, *_ = np.linalg.lstsq(phi, y_target, rcond=None)

    a = theta[:na]
    b = theta[na:]
    den_z = np.concatenate([np.array([1.0]), a])
    num_z = np.concatenate([np.zeros(nk), b])

    y_hat = simulate_arx(u=u, den_z=den_z, num_z=num_z)
    return den_z, num_z, y_hat


def simulate_arx(u: np.ndarray, den_z: np.ndarray, num_z: np.ndarray) -> np.ndarray:
    na = den_z.shape[0] - 1
    nb_total = num_z.shape[0]
    n = u.shape[0]
    y_hat = np.zeros(n, dtype=float)

    for k in range(n):
        acc = 0.0
        for i in range(1, na + 1):
            if k - i >= 0:
                acc -= den_z[i] * y_hat[k - i]
        for j in range(nb_total):
            if k - j >= 0:
                acc += num_z[j] * u[k - j]
        y_hat[k] = acc

    return y_hat


def _poly_from_roots_real(roots: np.ndarray) -> np.ndarray:
    if roots.size == 0:
        return np.array([1.0])
    coeff = np.poly(roots)
    return np.real_if_close(coeff, tol=1_000).astype(float)


def discrete_to_continuous_matched(
    den_z: np.ndarray, num_z: np.ndarray, dt: float
) -> tuple[np.ndarray, np.ndarray]:
    if dt <= 0.0:
        raise ValueError("dt must be positive")

    # ARX form in z^-1 -> polynomial in q = z^-1. Convert to z-domain roots.
    den_q = np.asarray(den_z, dtype=float)
    num_q = np.asarray(num_z, dtype=float)
    num_q_nz = (
        num_q[np.argmax(np.abs(num_q) > 1e-12) :]
        if np.any(np.abs(num_q) > 1e-12)
        else np.array([0.0])
    )

    poles_q = np.roots(den_q) if den_q.size > 1 else np.array([])
    zeros_q = np.roots(num_q_nz) if num_q_nz.size > 1 else np.array([])

    poles_z = np.array([1.0 / p for p in poles_q if not np.isclose(p, 0.0)], dtype=complex)
    zeros_z = np.array([1.0 / z for z in zeros_q if not np.isclose(z, 0.0)], dtype=complex)

    poles_s = np.log(poles_z) / dt if poles_z.size else np.array([])
    zeros_s = np.log(zeros_z) / dt if zeros_z.size else np.array([])

    den_s_base = _poly_from_roots_real(poles_s)
    num_s_base = _poly_from_roots_real(zeros_s)

    g0_d = float(np.sum(num_z) / np.sum(den_z)) if not np.isclose(np.sum(den_z), 0.0) else 0.0
    g0_c_base = (
        float(num_s_base[-1] / den_s_base[-1]) if den_s_base.size and den_s_base[-1] != 0.0 else 1.0
    )
    gain_scale = 1.0 if np.isclose(g0_c_base, 0.0) else g0_d / g0_c_base
    num_s = gain_scale * num_s_base
    den_s = den_s_base
    return num_s.astype(float), den_s.astype(float)


def tf_string_z(num_z: np.ndarray, den_z: np.ndarray) -> str:
    def _poly_to_str(coeff: np.ndarray, var: str) -> str:
        parts: list[str] = []
        for i, c in enumerate(coeff):
            if np.isclose(c, 0.0):
                continue
            term = f"{c:.6g}"
            if i > 0:
                term += f" {var}^{-i}"
            parts.append(term)
        return " + ".join(parts) if parts else "0"

    num = _poly_to_str(num_z, "z")
    den = _poly_to_str(den_z, "z")
    return f"G(z) = ({num}) / ({den})"


def tf_string_s(num_s: np.ndarray, den_s: np.ndarray) -> str:
    def _poly_to_desc(coeff: np.ndarray, var: str) -> str:
        order = coeff.shape[0] - 1
        parts: list[str] = []
        for i, c in enumerate(coeff):
            p = order - i
            if np.isclose(c, 0.0):
                continue
            if p == 0:
                parts.append(f"{c:.6g}")
            elif p == 1:
                parts.append(f"{c:.6g} {var}")
            else:
                parts.append(f"{c:.6g} {var}^{p}")
        return " + ".join(parts) if parts else "0"

    num = _poly_to_desc(num_s, "s")
    den = _poly_to_desc(den_s, "s")
    return f"G(s) = ({num}) / ({den})"


def identify_per_step(
    control_path: str | Path,
    output_path: str | Path,
) -> list[dict]:
    """Identify G(s) and G(z) for each step region separately.

    Detects step transitions in u, then extracts a segment after each step
    and identifies the plant independently. Returns list of dicts with parameters.

    Useful for diagnosing nonlinearity: if K, ζ, ω_n vary across steps,
    the plant is (moderately) nonlinear.
    """
    u_signal = load_signal_file(control_path)
    y_signal = load_signal_file(output_path)
    t, u, y = align_signals(u_signal, y_signal)
    dt = infer_dt(t)

    # Detect all step transitions
    du = np.diff(u, prepend=u[0])
    step_indices = np.where(np.abs(du) > 1e-12)[0]
    if len(step_indices) == 0:
        raise ValueError("No step transitions found in control signal")

    results = []
    for step_num, step_idx in enumerate(step_indices):
        # Extract only the segment from THIS step to the NEXT step
        # (or end of signal if it's the last step)
        next_step_idx = step_indices[step_num + 1] if step_num + 1 < len(step_indices) else len(t)

        t_seg = t[step_idx:next_step_idx]
        u_seg = u[step_idx:next_step_idx]
        y_seg = y[step_idx:next_step_idx]

        # Subtract initial offset from u to restart at 0
        u_base = float(u_seg[0])
        u_seg_zero = u_seg - u_base

        # Subtract initial offset from y (pseudo-steady-state before step)
        y_base = float(y_seg[0])
        y_seg_zero = y_seg - y_base

        # Create temporary signal files for this segment
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as uf:
            for ti, ui in zip(t_seg, u_seg_zero):
                uf.write(f"{ti:.6f} {ui:.6f}\n")
            u_temp = uf.name

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as yf:
            for ti, yi in zip(t_seg, y_seg):
                yf.write(f"{ti:.6f} {yi:.6f}\n")
            y_temp = yf.name

        try:
            result, _, _, _ = identify_zoh_from_second_order(u_temp, y_temp)
            u_level = float(u[step_idx])  # Input level at this step
            y_initial = float(y[step_idx])  # Output level at this step

            results.append(
                {
                    "step_num": int(step_num),
                    "step_index": int(step_idx),
                    "u_level": float(u_level),
                    "y_initial": float(y_initial),
                    "time_at_step": float(t[step_idx]),
                    "segment_length": int(next_step_idx - step_idx),
                    "gain": float(result.gain) if result.gain is not None else None,
                    "zeta": float(result.zeta) if result.zeta is not None else None,
                    "omega_n": float(result.omega_n) if result.omega_n is not None else None,
                    "delay": float(result.delay) if result.delay is not None else None,
                    "fit_rmse": float(result.fit_rmse),
                    "num_s": result.num_s.tolist(),
                    "den_s": result.den_s.tolist(),
                    "num_z": result.num_z.tolist(),
                    "den_z": result.den_z.tolist(),
                }
            )
        except ValueError as e:
            # If bounds problem, try unbounded fitting with relaxed constraints
            try:
                from control_lab.ident.second_order import (
                    StepResponseData,
                    estimate_second_order_step_model,
                )

                step_index, step_amplitude = _detect_step(u_seg_zero)
                data = StepResponseData(
                    t=t_seg,
                    y=y_seg_zero,
                    u=u_seg_zero,
                    step_index=step_index,
                    step_amplitude=step_amplitude,
                )
                model = estimate_second_order_step_model(data)
                y0_guess = float(
                    np.mean(y_seg_zero[: max(2, min(20, step_index if step_index > 0 else 2))])
                )

                def _simulate_with_params(params):
                    gain_p = float(params[0])
                    zeta_p = float(params[1])
                    omega_n_p = float(params[2])
                    delay_p = float(params[3])
                    y0_p = float(params[4])

                    num_s_p = np.array([gain_p * omega_n_p**2], dtype=float)
                    den_s_p = np.array([1.0, 2.0 * zeta_p * omega_n_p, omega_n_p**2], dtype=float)
                    num_z_p, den_z_p = continuous_to_discrete_zoh(
                        num_s=num_s_p, den_s=den_s_p, dt=dt
                    )
                    delay_samples_p = max(0, int(round(delay_p / dt)))
                    if delay_samples_p > 0:
                        num_z_p = np.concatenate([np.zeros(delay_samples_p, dtype=float), num_z_p])

                    y_hat_p = lfilter(num_z_p, den_z_p, u_seg_zero) + y0_p
                    return num_s_p, den_s_p, num_z_p, den_z_p, y_hat_p

                p0 = np.array(
                    [
                        max(1e-6, float(model.gain)),
                        float(np.clip(model.zeta, 0.05, 2.5)),
                        max(1e-4, float(model.omega_n)),
                        max(0.0, float(model.delay)),
                        y0_guess,
                    ],
                    dtype=float,
                )

                def _residuals(params):
                    _, _, _, _, y_hat_p = _simulate_with_params(params)
                    return y_hat_p - y_seg_zero

                opt = least_squares(_residuals, p0, method="trf", loss="soft_l1")
                p_opt = opt.x if opt.success else p0
                num_s, den_s, num_z, den_z, y_hat = _simulate_with_params(p_opt)
                gain_opt, zeta_opt, omega_n_opt, delay_opt, _ = [float(v) for v in p_opt]

                rmse = float(np.sqrt(np.mean((y_seg_zero - y_hat) ** 2)))

                results.append(
                    {
                        "step_num": int(step_num),
                        "step_index": int(step_idx),
                        "u_level": float(u[step_idx]),
                        "y_initial": float(y[step_idx]),
                        "time_at_step": float(t[step_idx]),
                        "segment_length": int(next_step_idx - step_idx),
                        "gain": float(gain_opt),
                        "zeta": float(zeta_opt),
                        "omega_n": float(omega_n_opt),
                        "delay": float(delay_opt),
                        "fit_rmse": float(rmse),
                        "num_s": num_s.tolist(),
                        "den_s": den_s.tolist(),
                        "num_z": num_z.tolist(),
                        "den_z": den_z.tolist(),
                        "fit_method": "unbounded_fallback",
                    }
                )
            except Exception as e2:
                print(f"Warning: Step {step_num} fitting failed: {e2}")
        finally:
            Path(u_temp).unlink()
            Path(y_temp).unlink()

    return results


def identify_zoh_models(
    control_path: str | Path,
    output_path: str | Path,
    na: int = 2,
    nb: int = 2,
    nk: int = 1,
) -> tuple[ZOHIdentificationResult, np.ndarray, np.ndarray, np.ndarray]:
    u_signal = load_signal_file(control_path)
    y_signal = load_signal_file(output_path)
    t, u, y = align_signals(u_signal, y_signal)
    dt = infer_dt(t)

    den_z, num_z, y_hat = estimate_arx(u=u, y=y, na=na, nb=nb, nk=nk)
    num_s, den_s = discrete_to_continuous_matched(den_z=den_z, num_z=num_z, dt=dt)

    rmse = float(np.sqrt(np.mean((y - y_hat) ** 2)))
    result = ZOHIdentificationResult(
        dt=dt,
        na=na,
        nb=nb,
        nk=nk,
        den_z=den_z,
        num_z=num_z,
        den_s=den_s,
        num_s=num_s,
        fit_rmse=rmse,
        y_hat=y_hat,
    )
    return result, t, u, y
