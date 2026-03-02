"""Discretisation helpers for continuous-time state-space models."""

from __future__ import annotations

import numpy as np
import scipy.signal


def discretize(
    A: np.ndarray, B: np.ndarray, dt: float, method: str = "zoh"
) -> tuple[np.ndarray, np.ndarray]:
    """Discretise (A, B) with the given method.

    Returns
    -------
    Ad, Bd : discrete-time system matrices.
    """
    n = A.shape[0]
    m = B.shape[1]
    C = np.eye(n)
    D = np.zeros((n, m))
    Ad, Bd, _, _, _ = scipy.signal.cont2discrete((A, B, C, D), dt, method=method)
    return Ad, Bd


def compare_methods(A: np.ndarray, B: np.ndarray, dt: float) -> dict:
    """Discretise (A, B) with ZOH, Euler (forward), and Tustin methods.

    Returns a dict keyed by method name, each value being a dict
    ``{'Ad': ..., 'Bd': ...}`` or ``{'error': <message>}``.
    """
    results: dict = {}
    for method in ("zoh", "euler", "tustin"):
        try:
            Ad, Bd = discretize(A, B, dt, method=method)
            results[method] = {"Ad": Ad, "Bd": Bd}
        except Exception as exc:  # noqa: BLE001
            results[method] = {"error": str(exc)}
    return results
