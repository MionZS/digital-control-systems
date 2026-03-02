"""Tests for LQR design functions and LQRController."""

from __future__ import annotations

import numpy as np

from control_lab.design.lqr import LQRController, lqr_continuous, lqr_discrete
from control_lab.models.lti import LTIModel


def test_lqr_continuous():
    model = LTIModel.mass_spring_damper(m=1.0, k=1.0, c=0.5)
    Q = np.eye(2)
    R = np.eye(1)
    K, P, E = lqr_continuous(model.A, model.B, Q, R)

    assert np.all(np.isfinite(K)), "K contains non-finite values"
    assert np.all(np.isfinite(P)), "P contains non-finite values"

    A_cl = model.A - model.B @ K
    eigs = np.linalg.eigvals(A_cl)
    assert np.all(np.real(eigs) < 0), f"Closed-loop is not stable: eigenvalues = {eigs}"


def test_lqr_discrete():
    model = LTIModel.mass_spring_damper(m=1.0, k=1.0, c=0.5, dt=0.01)
    Q = np.eye(2)
    R = np.eye(1)
    K, P, E = lqr_discrete(model.A, model.B, Q, R)

    assert np.all(np.isfinite(K)), "K contains non-finite values"
    assert np.all(np.isfinite(P)), "P contains non-finite values"

    A_cl = model.A - model.B @ K
    eigs = np.linalg.eigvals(A_cl)
    assert np.all(np.abs(eigs) < 1.0), (
        f"Discrete closed-loop is not stable: |eigs| = {np.abs(eigs)}"
    )


def test_lqr_controller():
    model = LTIModel.mass_spring_damper(m=1.0, k=1.0, c=0.5)
    Q = np.eye(2)
    R = np.eye(1)
    K, _, _ = lqr_continuous(model.A, model.B, Q, R)

    ctrl = LQRController(K)
    x = np.array([1.0, 0.0])
    r = np.array([0.0, 0.0])
    u = ctrl.compute(x, r, 0.0)

    assert u.shape == (1,), f"Expected shape (1,), got {u.shape}"
    assert np.all(np.isfinite(u)), "Control output contains non-finite values"
