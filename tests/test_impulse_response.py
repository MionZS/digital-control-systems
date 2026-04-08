from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from control_lab.ident.impulse_response import (
    identify_fir_from_impulse,
    impulse_summary,
    load_impulse_response_csv,
    load_impulse_response_data,
    load_impulse_response_txt,
)


def test_load_impulse_response_csv_reads_time_and_output():
    csv_path = Path("input/impulse_response/sample_impulse.csv")
    t, y = load_impulse_response_csv(csv_path)
    assert t.ndim == 1
    assert y.ndim == 1
    assert t.shape[0] == y.shape[0]
    assert t.shape[0] > 3


def test_load_impulse_response_txt_estimates_impulse_from_step_data():
    txt_path = Path("input/impulse_response/sample_thermal_step.txt")
    t, h = load_impulse_response_txt(txt_path)
    assert t.ndim == 1
    assert h.ndim == 1
    assert t.shape[0] == h.shape[0]
    assert np.max(np.abs(h)) > 0.0


def test_load_impulse_response_data_supports_txt_and_csv():
    t_csv, y_csv = load_impulse_response_data("input/impulse_response/sample_impulse.csv")
    t_txt, h_txt = load_impulse_response_data("input/impulse_response/sample_thermal_step.txt")
    assert t_csv.shape[0] == y_csv.shape[0]
    assert t_txt.shape[0] == h_txt.shape[0]


def test_identify_fir_from_impulse_uses_impulse_as_numerator():
    t = np.array([0.0, 0.1, 0.2, 0.3])
    y = np.array([0.0, 1.0, 0.5, 0.25])
    model = identify_fir_from_impulse(t, y)
    np.testing.assert_allclose(model.numerator, y)
    np.testing.assert_allclose(model.denominator, np.array([1.0]))


def test_impulse_summary_has_expected_fields():
    t = np.array([0.0, 0.1, 0.2, 0.3])
    y = np.array([0.0, 1.0, 0.5, 0.25])
    model = identify_fir_from_impulse(t, y)
    summary = impulse_summary(model)
    assert set(summary.keys()) == {
        "dt",
        "num_taps",
        "dc_gain",
        "peak_amplitude",
        "peak_index",
    }
    assert summary["peak_index"] == pytest.approx(1.0)
