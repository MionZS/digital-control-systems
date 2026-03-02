"""Nonlinear model definition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np


@dataclass
class NonlinearModel:
    """Nonlinear state-space model.

    xdot = f(x, u, t)
    y    = g(x, u, t)
    """

    f_func: Callable[[np.ndarray, np.ndarray, float], np.ndarray]
    g_func: Callable[[np.ndarray, np.ndarray, float], np.ndarray]
    n_states: int
    n_inputs: int
    n_outputs: int
    dt: Optional[float] = None

    @classmethod
    def inverted_pendulum(
        cls, m: float, l: float, g: float = 9.81, b: float = 0.1
    ) -> "NonlinearModel":
        """Inverted pendulum model.

        State  = [theta, theta_dot]  (theta=0 at unstable upright equilibrium)
        Input  = torque (Nm)
        Output = full state
        """
        _g, _m, _l, _b = g, m, l, b

        def f(x: np.ndarray, u: np.ndarray, t: float) -> np.ndarray:
            theta, dtheta = float(x[0]), float(x[1])
            tau = float(u[0])
            ddtheta = (_m * _g * _l * np.sin(theta) - _b * dtheta + tau) / (_m * _l**2)
            return np.array([dtheta, ddtheta])

        def g_out(x: np.ndarray, u: np.ndarray, t: float) -> np.ndarray:
            return x.copy()

        return cls(f_func=f, g_func=g_out, n_states=2, n_inputs=1, n_outputs=2)
