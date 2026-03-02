"""LQR design functions and LQR controller class."""

from __future__ import annotations

import numpy as np
import scipy.linalg


def lqr_continuous(
    A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve the continuous-time LQR problem.

    Minimises  ∫ (x^T Q x + u^T R u) dt  subject to  xdot = Ax + Bu.

    Returns
    -------
    K : gain matrix  (u = -K x)
    P : solution of the continuous algebraic Riccati equation
    E : closed-loop eigenvalues of (A - B K)
    """
    P = scipy.linalg.solve_continuous_are(A, B, Q, R)
    K = np.linalg.solve(R, B.T @ P)
    E = np.linalg.eigvals(A - B @ K)
    return K, P, E


def lqr_discrete(
    A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve the discrete-time LQR problem.

    Minimises  Σ (x^T Q x + u^T R u)  subject to  x[k+1] = A x[k] + B u[k].

    Returns
    -------
    K : gain matrix  (u[k] = -K x[k])
    P : solution of the discrete algebraic Riccati equation
    E : closed-loop eigenvalues of (A - B K)
    """
    P = scipy.linalg.solve_discrete_are(A, B, Q, R)
    K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    E = np.linalg.eigvals(A - B @ K)
    return K, P, E


class LQRController:
    """State-feedback LQR controller.

    Implements ``ControllerProtocol``.  Computes  u = -K (x - r).
    """

    def __init__(self, K: np.ndarray) -> None:
        self.K = np.asarray(K, dtype=float)

    def compute(self, x: np.ndarray, r: np.ndarray, t: float) -> np.ndarray:  # noqa: ARG002
        return -(self.K @ (x - r))

    def reset(self) -> None:
        pass
