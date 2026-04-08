from __future__ import annotations

from pathlib import Path

import numpy as np

from control_lab.ident.zoh_ident import (
    estimate_arx,
    identify_zoh_from_second_order,
    load_signal_file,
    tf_string_s,
    tf_string_z,
)


def test_estimate_arx_recovers_simple_first_order():
    # y[k] = 0.8 y[k-1] + 0.2 u[k-1]
    n = 400
    u = np.zeros(n)
    u[20:] = 1.0
    y = np.zeros(n)
    for k in range(1, n):
        y[k] = 0.8 * y[k - 1] + 0.2 * u[k - 1]

    den_z, num_z, y_hat = estimate_arx(u=u, y=y, na=1, nb=1, nk=1)

    assert np.isclose(den_z[1], -0.8, atol=1e-2)
    assert np.isclose(num_z[1], 0.2, atol=1e-2)
    assert np.sqrt(np.mean((y - y_hat) ** 2)) < 1e-2


def test_tf_string_helpers_return_expected_prefixes():
    gz = tf_string_z(np.array([0.0, 0.2]), np.array([1.0, -0.8]))
    gs = tf_string_s(np.array([1.0]), np.array([1.0, 2.0, 1.0]))
    assert gz.startswith("G(z) =")
    assert gs.startswith("G(s) =")


def test_load_signal_file_parses_lvm(tmp_path: Path):
    lvm = tmp_path / "sig.lvm"
    lvm.write_text(
        "LabVIEW Measurement\n***End_of_Header***\n0.0\t0.0\n0.1\t1.0\n0.2\t2.0\n",
        encoding="utf-8",
    )

    signal = load_signal_file(lvm)
    assert signal.t.shape[0] == 3
    assert np.isclose(signal.v[-1], 2.0)


def test_load_signal_file_parses_lvm_decimal_comma(tmp_path: Path):
    lvm = tmp_path / "sig_comma.lvm"
    lvm.write_text(
        "LabVIEW Measurement\n"
        "***End_of_Header***\n"
        "X_Value\tUntitled\n"
        "0,0\t0,0\n"
        "1,0\t1,5\n"
        "2,0\t3,0\n",
        encoding="utf-8",
    )

    signal = load_signal_file(lvm)
    assert signal.t.shape[0] == 3
    assert np.isclose(signal.t[1], 1.0)
    assert np.isclose(signal.v[1], 1.5)


def test_identify_zoh_from_second_order_returns_gs_and_gz(tmp_path: Path):
    t = np.arange(0.0, 20.0, 0.1)
    u = np.zeros_like(t)
    u[t >= 2.0] = 1.0

    # Lightly underdamped step-like output.
    zeta = 0.5
    omega_n = 1.0
    tau = np.maximum(0.0, t - 2.0)
    omega_d = omega_n * np.sqrt(1.0 - zeta**2)
    phi = np.arctan(np.sqrt(1.0 - zeta**2) / zeta)
    y = 1.2 * (
        1.0 - np.exp(-zeta * omega_n * tau) / np.sqrt(1.0 - zeta**2) * np.sin(omega_d * tau + phi)
    )

    ctrl = tmp_path / "control.csv"
    out = tmp_path / "output.csv"
    ctrl.write_text("t,u\n" + "\n".join(f"{ti},{ui}" for ti, ui in zip(t, u)), encoding="utf-8")
    out.write_text("t,y\n" + "\n".join(f"{ti},{yi}" for ti, yi in zip(t, y)), encoding="utf-8")

    result, t_aligned, u_aligned, y_aligned = identify_zoh_from_second_order(ctrl, out)
    assert result.num_s.size >= 1
    assert result.den_s.size == 3
    assert result.num_z.size >= 1
    assert result.den_z.size >= 1
    assert result.method == "second-order-zoh-step-invariant"
    assert t_aligned.shape == u_aligned.shape == y_aligned.shape
