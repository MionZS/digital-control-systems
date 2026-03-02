"""Excitation signal generators for system identification."""

from __future__ import annotations

import numpy as np


def generate_prbs(
    n_steps: int, dt: float, amplitude: float = 1.0, seed: int = 42
) -> dict:
    """Generate a pseudo-random binary sequence (PRBS).

    Returns a dict with keys ``'t'`` and ``'u'``.
    """
    rng = np.random.default_rng(seed)
    u = rng.choice(np.array([-amplitude, amplitude]), size=n_steps)
    t = np.arange(n_steps) * dt
    return {"t": t, "u": u}


def generate_multisine(
    freqs: list[float],
    amplitudes: list[float],
    t_grid: np.ndarray,
    seed: int = 42,
) -> np.ndarray:
    """Sum of sinusoids with random phase offsets.

    Returns a 1-D array of length ``len(t_grid)``.
    """
    rng = np.random.default_rng(seed)
    u = np.zeros_like(t_grid, dtype=float)
    for freq, amp in zip(freqs, amplitudes):
        phase = rng.uniform(0.0, 2.0 * np.pi)
        u += amp * np.sin(2.0 * np.pi * freq * t_grid + phase)
    return u
