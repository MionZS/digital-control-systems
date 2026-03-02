"""Observer design: Luenberger and Kalman."""

from __future__ import annotations

import numpy as np
import scipy.linalg
import scipy.signal


def luenberger_gain(
    A: np.ndarray, C: np.ndarray, desired_poles: np.ndarray
) -> np.ndarray:
    """Compute the Luenberger observer gain via pole placement.

    Returns L such that eigenvalues of (A - L C) equal *desired_poles*.
    """
    result = scipy.signal.place_poles(A.T, C.T, desired_poles)
    return result.gain_matrix.T


def kalman_gain(
    A: np.ndarray,
    C: np.ndarray,
    Q_noise: np.ndarray,
    R_noise: np.ndarray,
) -> np.ndarray:
    """Compute the steady-state Kalman gain for a continuous-time system.

    Solves the observer (filter) Riccati equation and returns L = P C^T R^{-1}.
    """
    # Observer Riccati:  A P + P A^T - P C^T R^{-1} C P + Q = 0
    # scipy.linalg.solve_continuous_are(A, B, Q, R) solves A^T X + X A - X B R^{-1} B^T X + Q = 0
    # Transpose trick: pass A.T, C.T to recover observer solution
    P = scipy.linalg.solve_continuous_are(A.T, C.T, Q_noise, R_noise)
    L = P @ C.T @ np.linalg.inv(R_noise)
    return L


class StateObserver:
    """Continuous-time Luenberger / Kalman state observer (Euler integration).

    Update rule (discretised by Euler):
        x̂_dot = A x̂ + B u + L (y - C x̂)
        x̂[k+1] = x̂[k] + dt * x̂_dot
    """

    def __init__(self, A: np.ndarray, B: np.ndarray, C: np.ndarray, L: np.ndarray) -> None:
        self.A = np.asarray(A, dtype=float)
        self.B = np.asarray(B, dtype=float)
        self.C = np.asarray(C, dtype=float)
        self.L = np.asarray(L, dtype=float)
        self.x_hat = np.zeros(self.A.shape[0])

    def update(self, y: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
        """Propagate the estimate one step and return the updated estimate."""
        y = np.atleast_1d(y)
        u = np.atleast_1d(u)
        innovation = y - self.C @ self.x_hat
        x_dot = self.A @ self.x_hat + self.B @ u + self.L @ innovation
        self.x_hat = self.x_hat + dt * x_dot
        return self.x_hat.copy()

    def reset(self) -> None:
        """Reset estimate to zero."""
        self.x_hat[:] = 0.0
