"""control-lab: modular control-systems simulation framework."""

from __future__ import annotations

__version__ = "0.1.0"

from control_lab.design.lqr import LQRController, lqr_continuous, lqr_discrete
from control_lab.design.pid import PIDController
from control_lab.models.lti import LTIModel
from control_lab.models.nonlinear import NonlinearModel
from control_lab.sim.backend_control import ControlBackend
from control_lab.sim.common import SimulationResult, compute_metrics

__all__ = [
    "__version__",
    "LTIModel",
    "NonlinearModel",
    "PIDController",
    "LQRController",
    "lqr_continuous",
    "lqr_discrete",
    "SimulationResult",
    "compute_metrics",
    "ControlBackend",
]
