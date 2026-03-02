"""design sub-package."""

from control_lab.design.discretization import compare_methods, discretize
from control_lab.design.lqr import LQRController, lqr_continuous, lqr_discrete
from control_lab.design.observers import StateObserver, kalman_gain, luenberger_gain
from control_lab.design.pid import PIDController

__all__ = [
    "PIDController",
    "LQRController",
    "lqr_continuous",
    "lqr_discrete",
    "StateObserver",
    "luenberger_gain",
    "kalman_gain",
    "discretize",
    "compare_methods",
]
