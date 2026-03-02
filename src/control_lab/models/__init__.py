"""models sub-package."""

from control_lab.models.datasets import generate_multisine, generate_prbs
from control_lab.models.lti import LTIModel
from control_lab.models.nonlinear import NonlinearModel

__all__ = [
    "LTIModel",
    "NonlinearModel",
    "generate_prbs",
    "generate_multisine",
]
