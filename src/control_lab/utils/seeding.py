"""Global random-seed helper."""

from __future__ import annotations

import random

import numpy as np


def set_global_seed(seed: int) -> None:
    """Set numpy and stdlib random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
