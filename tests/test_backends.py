"""Tests for simulation backends and metrics (skipped without python-control)."""

from __future__ import annotations

import numpy as np
import pytest

# Skip entire module if python-control is not installed
control = pytest.importorskip("control")

from control_lab.design.pid import PIDController  # noqa: E402
from control_lab.models.lti import LTIModel  # noqa: E402
from control_lab.models.nonlinear import NonlinearModel  # noqa: E402
from control_lab.sim.backend_control import ControlBackend  # noqa: E402
from control_lab.sim.common import compute_metrics  # noqa: E402


def test_backend_control_lti():
    model = LTIModel.mass_spring_damper(m=1.0, k=1.0, c=0.5)
    pid = PIDController(kp=10.0, ki=1.0, kd=2.0, dt=0.01)
    r_func = lambda t: np.array([1.0])  # noqa: E731
    backend = ControlBackend()
    result = backend.simulate(model, pid, np.array([0.0, 0.0]), r_func, (0.0, 2.0), 0.01)

    n = result.t.shape[0]
    assert n > 0, "Simulation produced no steps"
    assert result.x.shape == (n, 2), f"Unexpected x shape: {result.x.shape}"
    assert result.y.shape == (n, 1), f"Unexpected y shape: {result.y.shape}"
    assert result.u.shape == (n, 1), f"Unexpected u shape: {result.u.shape}"
    assert np.all(np.isfinite(result.x)), "State trajectory contains non-finite values"


def test_backend_control_nonlinear():
    model = NonlinearModel.inverted_pendulum(m=0.5, l=0.5)
    pid = PIDController(kp=50.0, ki=1.0, kd=5.0, dt=0.01)
    r_func = lambda t: np.zeros(2)  # noqa: E731
    backend = ControlBackend()
    result = backend.simulate(
        model, pid, np.array([0.1, 0.0]), r_func, (0.0, 1.0), 0.01
    )
    assert np.all(np.isfinite(result.x)), "Nonlinear trajectory contains non-finite values"


def test_metrics():
    model = LTIModel.mass_spring_damper(m=1.0, k=1.0, c=0.5)
    pid = PIDController(kp=10.0, ki=1.0, kd=2.0, dt=0.01)
    r_func = lambda t: np.array([1.0])  # noqa: E731
    backend = ControlBackend()
    result = backend.simulate(model, pid, np.array([0.0, 0.0]), r_func, (0.0, 5.0), 0.01)
    metrics = compute_metrics(result, r_final=1.0)

    required_keys = {"overshoot", "settling_time", "iae", "ise", "control_effort"}
    assert required_keys.issubset(metrics.keys()), f"Missing metric keys: {required_keys - metrics.keys()}"
    for key, val in metrics.items():
        assert np.isfinite(val) or val == float("inf"), f"Metric {key} = {val} is unexpected"
