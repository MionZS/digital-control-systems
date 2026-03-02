"""Tests for LTIModel."""

from __future__ import annotations

import numpy as np
import pytest

from control_lab.models.lti import LTIModel


def test_lti_creation():
    model = LTIModel.mass_spring_damper(m=1.0, k=1.0, c=0.5)
    assert model.A.shape == (2, 2)
    assert model.B.shape == (2, 1)
    assert model.C.shape == (1, 2)
    assert model.D.shape == (1, 1)
    assert model.n_states == 2
    assert model.n_inputs == 1
    assert model.n_outputs == 1
    assert model.dt is None


def test_lti_discretize():
    model = LTIModel.mass_spring_damper(m=1.0, k=1.0, c=0.5)
    dt = 0.01
    disc = model.to_discrete_time(dt)
    assert disc.dt == pytest.approx(dt)
    assert disc.is_stable(), "Discrete MSD model should be stable"


def test_lti_stability():
    model = LTIModel.mass_spring_damper(m=1.0, k=1.0, c=0.5)
    assert model.is_stable(), "MSD with positive damping should be stable"

    # Unstable system: positive feedback
    A_unstable = np.array([[1.0]])
    B = np.array([[1.0]])
    C = np.array([[1.0]])
    D = np.array([[0.0]])
    unstable = LTIModel(A_unstable, B, C, D)
    assert not unstable.is_stable()


def test_lti_poles():
    A = np.array([[-2.0]])
    B = np.array([[1.0]])
    C = np.array([[1.0]])
    D = np.array([[0.0]])
    model = LTIModel(A, B, C, D)
    poles = model.poles()
    assert np.allclose(poles, [-2.0])
