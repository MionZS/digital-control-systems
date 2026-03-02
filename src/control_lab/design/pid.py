"""PID controller with anti-windup and derivative filter."""

from __future__ import annotations

import math
from typing import Union

import numpy as np


class PIDController:
    """Discrete-time PID controller with clamping anti-windup and first-order
    derivative filter.

    Control law (per sample):
        e[k]      = r[k] - y[k]
        D[k]      = (kd * (e[k] - e[k-1]) + tau * D[k-1]) / (tau + dt)
        u_raw[k]  = kp * e[k] + ki * integral[k] + D[k]
        u[k]      = clip(u_raw[k], u_min, u_max)
        integral[k+1] = integral[k] + e[k]*dt   (only when not saturated)
    """

    def __init__(
        self,
        kp: float,
        ki: float,
        kd: float,
        dt: float,
        u_min: float = -math.inf,
        u_max: float = math.inf,
        tau: float = 0.1,
    ) -> None:
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.u_min = u_min
        self.u_max = u_max
        self.tau = tau

        self._integral: float = 0.0
        self._prev_error: float = 0.0
        self._d_filtered: float = 0.0

    def compute(
        self,
        x: Union[np.ndarray, float],
        r: Union[np.ndarray, float],
        t: float,  # noqa: ARG002 — kept for ControllerProtocol compatibility
    ) -> np.ndarray:
        """Compute the control output.

        Parameters
        ----------
        x : measured output (scalar or array; first element used).
        r : reference / setpoint (scalar or array; first element used).
        t : current time (unused internally, required by protocol).
        """
        y = float(x[0]) if isinstance(x, np.ndarray) else float(x)
        ref = float(r[0]) if isinstance(r, np.ndarray) else float(r)

        error = ref - y

        # Filtered derivative
        self._d_filtered = (
            self.kd * (error - self._prev_error) + self.tau * self._d_filtered
        ) / (self.tau + self.dt)

        # Unsaturated output
        u_unsat = self.kp * error + self.ki * self._integral + self._d_filtered

        # Saturate
        u = float(np.clip(u_unsat, self.u_min, self.u_max))

        # Anti-windup: integrate only when not saturated
        if self.u_min <= u_unsat <= self.u_max:
            self._integral += error * self.dt

        self._prev_error = error
        return np.array([u])

    def reset(self) -> None:
        """Reset controller state."""
        self._integral = 0.0
        self._prev_error = 0.0
        self._d_filtered = 0.0

    @classmethod
    def tune_ziegler_nichols(cls, ku: float, tu: float, dt: float = 0.01) -> "PIDController":
        """Classic Ziegler-Nichols PID tuning from ultimate gain *ku* and period *tu*."""
        kp = 0.6 * ku
        ki = 2.0 * kp / tu
        kd = kp * tu / 8.0
        return cls(kp=kp, ki=ki, kd=kd, dt=dt)
