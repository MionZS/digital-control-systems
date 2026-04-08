"""Closed-loop thermal simulation using identified impulse-response model.

Controller tracks T_ref = T_out + delta_ref with PID and anti-windup.
Plant is simulated from the identified discrete transfer function G(z).

Example:
    uv run python Scripts/simulate_impulse_closed_loop.py \
      --model-json output/impulse_response/degrau1_lampada_26032026/identified_model.json \
      --disturbance-csv output/impulse_response/degrau1_lampada_26032026/degrau1_lampada_26032026.csv \
      --out-json output/impulse_response/closed_loop/closed_loop_results.json \
      --out-plot output/impulse_response/closed_loop/closed_loop_plot.png \
      --delta-ref 5.0
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_thermal_csv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows: list[tuple[float, float, float, float]] = []
    with path.open("r", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            try:
                t = float(row["t"])
                t_in = float(row["T_in"])
                u = float(row["u"])
                t_out = float(row["T_out"])
            except (KeyError, ValueError):
                continue
            rows.append((t, t_in, u, t_out))

    if not rows:
        raise ValueError(f"No valid rows found in {path}")

    arr = np.asarray(rows, dtype=float)
    return arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]


def estimate_default_pid(model_params: dict[str, float]) -> tuple[float, float, float]:
    # IMC-like PI for first-order process K/(tau s + 1), lambda=tau.
    k_process = float(model_params.get("K", 1.0))
    tau = float(model_params.get("tau", 30.0))
    if k_process <= 1e-9:
        k_process = 1.0
    if tau <= 1e-9:
        tau = 30.0

    kp = 1.0 / k_process
    ki = kp / tau
    kd = 0.0
    return kp, ki, kd


def infer_delay_samples(model_params: dict[str, float], dt: float) -> int:
    if "delay_samples" in model_params:
        return max(0, int(round(float(model_params["delay_samples"]))))
    if "delay" in model_params:
        return max(0, int(round(float(model_params["delay"]) / dt)))
    return 0


def simulate_closed_loop(
    t: np.ndarray,
    t_out: np.ndarray,
    b: np.ndarray,
    a: np.ndarray,
    delay_samples: int,
    delta_ref: float,
    kp: float,
    ki: float,
    kd: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(t)
    dt = float(np.median(np.diff(t)))

    u_cmd = np.zeros(n, dtype=float)
    y_rel = np.zeros(n, dtype=float)
    t_in = np.zeros(n, dtype=float)
    t_ref = t_out + float(delta_ref)

    integral = 0.0
    prev_error = float(t_ref[0] - t_out[0])

    na = len(a) - 1
    nb = len(b) - 1

    for k in range(n):
        if k == 0:
            t_in[k] = float(t_out[k])
        else:
            t_in[k] = float(t_out[k] + y_rel[k - 1])

        error = float(t_ref[k] - t_in[k])
        d_error = (error - prev_error) / dt if k > 0 else 0.0

        i_candidate = integral + ki * error * dt
        u_unsat = kp * error + i_candidate + kd * d_error
        u_sat = float(np.clip(u_unsat, 0.0, 1.0))

        at_upper = u_sat >= 1.0 - 1e-12 and error > 0.0
        at_lower = u_sat <= 1e-12 and error < 0.0

        if at_upper or at_lower:
            u_unsat_no_int = kp * error + integral + kd * d_error
            u_sat = float(np.clip(u_unsat_no_int, 0.0, 1.0))
        else:
            integral = i_candidate

        u_cmd[k] = u_sat

        yk = 0.0
        for j in range(nb + 1):
            if k - j < 0:
                continue
            u_eff_j = u_cmd[k - j - delay_samples] if k - j - delay_samples >= 0 else 0.0
            yk += float(b[j]) * float(u_eff_j)
        for i in range(1, na + 1):
            if k - i < 0:
                continue
            yk -= float(a[i]) * float(y_rel[k - i])

        y_rel[k] = yk
        prev_error = error

    t_in = t_out + y_rel
    return t_ref, t_in, u_cmd, y_rel


def save_plot(
    out_plot: Path,
    t: np.ndarray,
    t_out: np.ndarray,
    t_ref: np.ndarray,
    t_in: np.ndarray,
    u_cmd: np.ndarray,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)

    axes[0].plot(t, t_out, color="tab:gray", linewidth=1.2, label="T_out")
    axes[0].plot(
        t, t_ref, color="tab:orange", linewidth=1.2, linestyle="--", label="T_ref = T_out + 5"
    )
    axes[0].plot(t, t_in, color="tab:red", linewidth=1.2, label="T_in (closed-loop)")
    axes[0].set_title("Closed-loop thermal response")
    axes[0].set_xlabel("Time [s]")
    axes[0].set_ylabel("Temperature [C]")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best")

    axes[1].plot(t, 100.0 * u_cmd, color="tab:blue", linewidth=1.2, label="Control output [%]")
    axes[1].set_title("Controller output with saturation")
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("u [%]")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best")

    out_plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_plot, dpi=140)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Closed-loop thermal simulation from identified model"
    )
    parser.add_argument("--model-json", type=Path, required=True)
    parser.add_argument("--disturbance-csv", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-plot", type=Path, required=True)
    parser.add_argument("--delta-ref", type=float, default=5.0)
    parser.add_argument("--kp", type=float, default=None)
    parser.add_argument("--ki", type=float, default=None)
    parser.add_argument("--kd", type=float, default=0.0)
    args = parser.parse_args()

    model_data = json.loads(args.model_json.read_text(encoding="utf-8"))
    model = model_data.get("model", {})
    params = model.get("parameters", {}) if isinstance(model, dict) else {}
    gz = model.get("discrete_z_domain", {}) if isinstance(model, dict) else {}

    b = np.asarray(gz.get("numerator", []), dtype=float)
    a = np.asarray(gz.get("denominator", []), dtype=float)
    if b.size == 0 or a.size == 0:
        raise ValueError("Model JSON is missing discrete_z_domain coefficients")

    t, _, _, t_out = read_thermal_csv(args.disturbance_csv)
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 1.0
    delay_samples = infer_delay_samples(params if isinstance(params, dict) else {}, dt)

    if args.kp is None or args.ki is None:
        kp_default, ki_default, kd_default = estimate_default_pid(
            params if isinstance(params, dict) else {}
        )
        kp = kp_default if args.kp is None else float(args.kp)
        ki = ki_default if args.ki is None else float(args.ki)
        kd = kd_default if args.kd is None else float(args.kd)
    else:
        kp = float(args.kp)
        ki = float(args.ki)
        kd = float(args.kd)

    t_ref, t_in, u_cmd, y_rel = simulate_closed_loop(
        t=t,
        t_out=t_out,
        b=b,
        a=a,
        delay_samples=delay_samples,
        delta_ref=float(args.delta_ref),
        kp=kp,
        ki=ki,
        kd=kd,
    )

    e = t_ref - t_in
    payload = {
        "inputs": {
            "model_json": str(args.model_json),
            "disturbance_csv": str(args.disturbance_csv),
            "delta_ref": float(args.delta_ref),
        },
        "controller": {
            "kp": kp,
            "ki": ki,
            "kd": kd,
            "saturation": [0.0, 1.0],
            "anti_windup": "freeze integrator when saturated in same error direction",
        },
        "model": {
            "name": model.get("name") if isinstance(model, dict) else None,
            "order": model.get("order") if isinstance(model, dict) else None,
            "delay_samples": int(delay_samples),
            "gz_num": b.tolist(),
            "gz_den": a.tolist(),
        },
        "metrics": {
            "rmse_tracking": float(np.sqrt(np.mean(e**2))),
            "mae_tracking": float(np.mean(np.abs(e))),
            "max_abs_tracking_error": float(np.max(np.abs(e))),
            "u_mean_percent": float(np.mean(100.0 * u_cmd)),
            "u_max_percent": float(np.max(100.0 * u_cmd)),
            "u_sat_100_percent_ratio": float(np.mean(u_cmd >= 1.0 - 1e-12)),
        },
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    save_plot(args.out_plot, t, t_out, t_ref, t_in, u_cmd)

    print("Closed-loop simulation completed")
    print(f"- out_json: {args.out_json}")
    print(f"- out_plot: {args.out_plot}")
    print(f"- rmse_tracking: {payload['metrics']['rmse_tracking']:.3f}")
    print(f"- saturation_ratio_100: {payload['metrics']['u_sat_100_percent_ratio']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
