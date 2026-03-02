"""Thin wrappers around pysindy feature libraries.

pysindy is imported lazily inside each function so that the module is safe
to import even when pysindy is not installed.
"""

from __future__ import annotations


def PolynomialLibrary(degree: int = 2):
    """Return a pysindy PolynomialLibrary of the requested *degree*."""
    import pysindy as ps  # noqa: PLC0415

    return ps.PolynomialLibrary(degree=degree)


def FourierLibrary(n_freqs: int = 3):
    """Return a pysindy FourierLibrary with *n_freqs* frequencies."""
    import pysindy as ps  # noqa: PLC0415

    return ps.FourierLibrary(n_frequencies=n_freqs)
