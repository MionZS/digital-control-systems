from __future__ import annotations

import numpy as np

from control_lab.ident.second_order import (
    StepResponseData,
    estimate_second_order_step_model,
    second_order_summary,
)


def _second_order_step_response(
    t: np.ndarray,
    gain: float,
    zeta: float,
    omega_n: float,
    delay: float = 0.0,
) -> np.ndarray:
    response = np.zeros_like(t)
    tau = t - delay
    mask = tau >= 0.0
    if not np.any(mask):
        return response

    omega_d = omega_n * np.sqrt(1.0 - zeta**2)
    phi = np.arctan(np.sqrt(1.0 - zeta**2) / zeta)
    response[mask] = gain * (
        1.0
        - np.exp(-zeta * omega_n * tau[mask])
        / np.sqrt(1.0 - zeta**2)
        * np.sin(omega_d * tau[mask] + phi)
    )
    return response


def test_estimate_second_order_step_model_recovers_parameters():
    t = np.arange(0.0, 20.0, 0.01)
    gain = 2.0
    zeta = 0.35
    omega_n = 1.8
    delay = 0.4
    y = _second_order_step_response(t, gain=gain, zeta=zeta, omega_n=omega_n, delay=delay)

    model = estimate_second_order_step_model(StepResponseData(t=t, y=y, step_amplitude=1.0))
    summary = second_order_summary(model)

    assert np.isclose(summary["gain"], gain, rtol=0.1)
    assert np.isclose(summary["zeta"], zeta, rtol=0.2)
    assert np.isclose(summary["omega_n"], omega_n, rtol=0.2)
    assert np.isclose(summary["delay"], delay, atol=0.05)
    assert summary["overshoot"] > 0.0


def test_second_order_polynomial_is_reported():
    t = np.arange(0.0, 10.0, 0.01)
    y = _second_order_step_response(t, gain=1.0, zeta=0.5, omega_n=2.0)
    model = estimate_second_order_step_model(StepResponseData(t=t, y=y, step_amplitude=1.0))
    assert "s^2" in model.polynomial
    assert "G(s)" in model.transfer_function
