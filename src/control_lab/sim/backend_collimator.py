"""Collimator simulation backend (optional dependency).

Import is gracefully guarded — importing this module never raises ImportError.
An error is raised only when *instantiating* CollimatorBackend without
pycollimator installed.
"""

from __future__ import annotations

try:
    import pycollimator  # noqa: F401

    _HAS_COLLIMATOR = True
except ImportError:
    _HAS_COLLIMATOR = False

from control_lab.sim.common import SimulationResult  # noqa: E402


class CollimatorBackend:
    """Simulation backend wrapping pycollimator (requires ``control-lab[collimator]``)."""

    def __init__(self) -> None:
        if not _HAS_COLLIMATOR:
            raise ImportError(
                "pycollimator is not installed. "
                "Install the optional dependency with:\n"
                "    pip install 'control-lab[collimator]'"
            )

    def simulate(
        self,
        model: object,
        controller: object,
        x0,
        r_func,
        t_span: tuple[float, float],
        dt: float,
    ) -> SimulationResult:
        raise NotImplementedError(
            "CollimatorBackend.simulate is not yet implemented."
        )
