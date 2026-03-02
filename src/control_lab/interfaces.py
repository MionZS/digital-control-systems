"""Protocol interfaces for the control-lab framework."""

from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class ModelProtocol(Protocol):
    """Protocol for state-space models."""

    @property
    def A(self) -> np.ndarray: ...

    @property
    def B(self) -> np.ndarray: ...

    @property
    def C(self) -> np.ndarray: ...

    @property
    def D(self) -> np.ndarray: ...

    @property
    def dt(self) -> Optional[float]: ...

    @property
    def n_states(self) -> int: ...

    @property
    def n_inputs(self) -> int: ...

    @property
    def n_outputs(self) -> int: ...


@runtime_checkable
class ControllerProtocol(Protocol):
    """Protocol for feedback controllers."""

    def compute(self, x: np.ndarray, r: np.ndarray, t: float) -> np.ndarray: ...

    def reset(self) -> None: ...


@runtime_checkable
class SimulatorBackendProtocol(Protocol):
    """Protocol for simulation backends."""

    def simulate(
        self,
        model: object,
        controller: object,
        x0: np.ndarray,
        r_func: object,
        t_span: tuple[float, float],
        dt: float,
    ) -> dict: ...


@runtime_checkable
class IdentifierProtocol(Protocol):
    """Protocol for system identification algorithms."""

    def fit(self, data: dict) -> None: ...

    def predict(self, x0: np.ndarray, u_seq: object, t_grid: np.ndarray) -> np.ndarray: ...
