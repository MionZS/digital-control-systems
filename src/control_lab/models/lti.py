"""LTI state-space model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import scipy.linalg
import scipy.signal


@dataclass
class LTIModel:
    """Linear Time-Invariant state-space model.

    Continuous:  xdot = A x + B u,  y = C x + D u
    Discrete:    x[k+1] = A x[k] + B u[k],  y[k] = C x[k] + D u[k]
    """

    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    D: np.ndarray
    dt: Optional[float] = None

    def __post_init__(self) -> None:
        self.A = np.atleast_2d(np.asarray(self.A, dtype=float))
        self.B = np.atleast_2d(np.asarray(self.B, dtype=float))
        self.C = np.atleast_2d(np.asarray(self.C, dtype=float))
        self.D = np.atleast_2d(np.asarray(self.D, dtype=float))

    @property
    def n_states(self) -> int:
        return self.A.shape[0]

    @property
    def n_inputs(self) -> int:
        return self.B.shape[1]

    @property
    def n_outputs(self) -> int:
        return self.C.shape[0]

    def poles(self) -> np.ndarray:
        """Return system poles (eigenvalues of A)."""
        return np.linalg.eigvals(self.A)

    def zeros(self) -> np.ndarray:
        """Return transmission zeros."""
        import control

        sys = control.ss(self.A, self.B, self.C, self.D)
        return np.array(control.zeros(sys))

    def is_stable(self) -> bool:
        """Return True if the system is asymptotically stable."""
        p = self.poles()
        if self.dt is None:
            return bool(np.all(np.real(p) < 0))
        return bool(np.all(np.abs(p) < 1.0))

    def transfer_function(self):
        """Return the transfer function representation (python-control)."""
        import control

        ss_sys = control.ss(self.A, self.B, self.C, self.D)
        return control.ss2tf(ss_sys)

    def to_discrete_time(self, dt: float, method: str = "zoh") -> "LTIModel":
        """Discretise the continuous-time model."""
        if self.dt is not None:
            raise ValueError("Model is already discrete.")
        Ad, Bd, Cd, Dd, _ = scipy.signal.cont2discrete(
            (self.A, self.B, self.C, self.D), dt, method=method
        )
        return LTIModel(Ad, Bd, Cd, Dd, dt=dt)

    def to_continuous_time(self, method: str = "zoh") -> "LTIModel":
        """Recover continuous-time model from discrete-time model (ZOH inverse)."""
        if self.dt is None:
            return self
        dt = self.dt
        Ac = scipy.linalg.logm(self.A).real / dt
        n = self.n_states
        try:
            # Solve (Ad - I) Bc = Ac @ Bd  =>  Bc = (Ad-I)^{-1} Ac Bd
            Bc = np.linalg.solve(self.A - np.eye(n), Ac @ self.B).real
        except np.linalg.LinAlgError:
            Bc = (self.B / dt).real
        return LTIModel(Ac, Bc, self.C.copy(), self.D.copy(), dt=None)

    @classmethod
    def mass_spring_damper(
        cls, m: float, k: float, c: float, dt: Optional[float] = None
    ) -> "LTIModel":
        """Mass-spring-damper: state = [position, velocity], input = force."""
        A = np.array([[0.0, 1.0], [-k / m, -c / m]])
        B = np.array([[0.0], [1.0 / m]])
        C = np.array([[1.0, 0.0]])
        D = np.array([[0.0]])
        model = cls(A, B, C, D, dt=None)
        if dt is not None:
            return model.to_discrete_time(dt)
        return model
