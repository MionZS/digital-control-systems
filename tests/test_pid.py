"""Tests for PIDController."""

from __future__ import annotations

import numpy as np
import pytest

from control_lab.design.pid import PIDController


def test_pid_basic():
    """PID step response should converge to reference."""
    dt = 0.01
    pid = PIDController(kp=10.0, ki=5.0, kd=1.0, dt=dt, u_min=-200.0, u_max=200.0)
    x = np.array([0.0])
    r = np.array([1.0])
    # Simple first-order plant: xdot = u - x
    for i in range(600):
        t = i * dt
        u = pid.compute(x, r, t)
        x = x + dt * (u - x)
    assert abs(float(x[0]) - 1.0) < 0.05, f"PID did not converge: final x = {x[0]:.4f}"


def test_pid_saturation():
    """Output must stay within [u_min, u_max]."""
    dt = 0.01
    pid = PIDController(kp=1000.0, ki=0.0, kd=0.0, dt=dt, u_min=-5.0, u_max=5.0)
    x = np.array([0.0])
    r = np.array([1.0])
    u = pid.compute(x, r, 0.0)
    assert float(u[0]) <= 5.0
    assert float(u[0]) >= -5.0


def test_pid_antiwindup():
    """Integral should not grow unboundedly when output is saturated."""
    dt = 0.01
    pid = PIDController(kp=1.0, ki=100.0, kd=0.0, dt=dt, u_min=-1.0, u_max=1.0)
    x = np.array([0.0])
    r = np.array([10.0])  # large reference → permanent saturation
    for i in range(1000):
        pid.compute(x, r, i * dt)
    # With anti-windup the integral should be bounded by ~(r / ki) order of magnitude
    assert abs(pid._integral) < 1e4, f"Integral runaway: {pid._integral:.2e}"


def test_pid_reset():
    """reset() clears all internal state."""
    dt = 0.01
    pid = PIDController(kp=1.0, ki=1.0, kd=0.5, dt=dt)
    x = np.array([0.0])
    r = np.array([1.0])
    for i in range(100):
        pid.compute(x, r, i * dt)
    assert pid._integral != 0.0  # should have accumulated
    pid.reset()
    assert pid._integral == pytest.approx(0.0)
    assert pid._prev_error == pytest.approx(0.0)
    assert pid._d_filtered == pytest.approx(0.0)
