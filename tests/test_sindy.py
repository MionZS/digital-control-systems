"""Tests for SINDy identifier (skipped when pysindy is not installed)."""

from __future__ import annotations

import numpy as np
import pytest

pysindy = pytest.importorskip("pysindy")

from control_lab.ident.sindy_fit import SINDyIdentifier  # noqa: E402
from control_lab.ident.sindy_validate import rollout_error  # noqa: E402


def _make_harmonic_data(n_steps: int = 1000, dt: float = 0.01) -> dict:
    """Generate harmonic oscillator trajectory: xdot = [[0,1],[-1,0]] x."""
    t = np.arange(n_steps) * dt
    x = np.zeros((n_steps, 2))
    x[0] = [1.0, 0.0]
    A = np.array([[0.0, 1.0], [-1.0, 0.0]])
    for i in range(n_steps - 1):
        x[i + 1] = x[i] + dt * (A @ x[i])
    return {"x": x, "t": t}


def test_sindy_fit_harmonic_oscillator():
    data = _make_harmonic_data()
    ident = SINDyIdentifier(dt=0.01)
    ident.fit(data)
    assert ident.model is not None, "Model should be set after fit()"


def test_sindy_predict_shape():
    data = _make_harmonic_data()
    ident = SINDyIdentifier(dt=0.01)
    ident.fit(data)
    t_grid = data["t"][:50]
    x0 = data["x"][0]
    x_pred = ident.predict(x0, None, t_grid)
    assert x_pred.ndim == 2, "predict() should return a 2-D array"
    assert x_pred.shape[1] == 2, f"Expected 2 states, got {x_pred.shape[1]}"


def test_rollout_error():
    data = _make_harmonic_data()
    ident = SINDyIdentifier(dt=0.01)
    ident.fit(data)
    err = rollout_error(ident, data)
    assert np.isfinite(err), f"rollout_error returned non-finite value: {err}"
