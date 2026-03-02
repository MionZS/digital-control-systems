"""Simulation result dataclass and performance metrics."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SimulationResult:
    """Container for a closed-loop simulation trajectory."""

    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    u: np.ndarray
    metadata: dict = field(default_factory=dict)


def compute_metrics(result: SimulationResult, r_final: float) -> dict:
    """Compute standard step-response performance metrics.

    Parameters
    ----------
    result  : simulation result
    r_final : steady-state reference value

    Returns
    -------
    dict with keys: overshoot, settling_time, iae, ise, control_effort
    """
    t = result.t
    y_arr = result.y[:, 0] if result.y.ndim > 1 else result.y.ravel()
    u_arr = result.u[:, 0] if result.u.ndim > 1 else result.u.ravel()

    # Overshoot (percent)
    if abs(r_final) > 1e-12:
        y_peak = np.max(y_arr) if r_final > 0 else np.min(y_arr)
        overshoot = max(0.0, (y_peak - r_final) / abs(r_final) * 100.0)
    else:
        overshoot = 0.0

    # Settling time: 2 % band (or fixed tolerance when r_final≈0)
    tol = 0.02 * abs(r_final) if abs(r_final) > 1e-6 else 0.02
    in_band = np.abs(y_arr - r_final) <= tol

    settling_time: float = float("inf")
    for i in range(len(t) - 1, -1, -1):
        if not in_band[i]:
            if i + 1 < len(t):
                settling_time = float(t[i + 1])
            break
    else:
        settling_time = float(t[0])

    error = y_arr - r_final
    _trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    iae = float(_trapz(np.abs(error), t))
    ise = float(_trapz(error**2, t))
    control_effort = float(_trapz(u_arr**2, t))

    return {
        "overshoot": float(overshoot),
        "settling_time": float(settling_time),
        "iae": iae,
        "ise": ise,
        "control_effort": control_effort,
    }
