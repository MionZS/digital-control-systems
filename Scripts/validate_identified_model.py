"""Validate an identified model against a new input/output experiment.

Example:
    uv run python Scripts/validate_identified_model.py \
        --model-json output/box4/identified_model.json \
        --control input/ValidacaoBox4/controle.lvm \
        --output input/ValidacaoBox4/saida.lvm \
        --out-json output/box4/validation_results.json \
        --out-plot output/box4/validation_plot.png \
        --time-scale 0.1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import lfilter


def read_lvm(path: Path) -> tuple[np.ndarray, np.ndarray]:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()

    end_idx = None
    for i, line in enumerate(lines):
        if "***End_of_Header***" in line:
            end_idx = i
            break
    if end_idx is None:
        msg = f"Could not find LVM header end in {path}"
        raise ValueError(msg)

    rows: list[list[float]] = []
    for line in lines[end_idx + 1 :]:
        parts = [p for p in line.strip().split("\t") if p != ""]
        if len(parts) < 2:
            continue
        try:
            vals = [float(p.replace(",", ".")) for p in parts[:2]]
            rows.append(vals)
        except ValueError:
            continue

    if not rows:
        msg = f"No numeric data found in {path}"
        raise ValueError(msg)

    arr = np.asarray(rows, dtype=float)
    return arr[:, 0], arr[:, 1]


def align_timebase(
    tu: np.ndarray,
    u: np.ndarray,
    ty: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    if len(tu) != len(ty):
        raise ValueError("Control and output files have different lengths.")
    if not np.allclose(tu, ty, atol=1e-9, rtol=0.0):
        raise ValueError("Control and output files do not share the same time base.")
    if len(tu) < 3:
        raise ValueError("Not enough samples.")
    dt = float(np.median(np.diff(tu)))
    return tu.copy(), u.copy(), y.copy(), dt


def simulate_discrete(
    u: np.ndarray,
    gz_num: np.ndarray,
    gz_den: np.ndarray,
    delay_samples: int,
) -> np.ndarray:
    y_model_raw = lfilter(gz_num, gz_den, u)
    if isinstance(y_model_raw, tuple):
        y_model = np.asarray(y_model_raw[0], dtype=float)
    else:
        y_model = np.asarray(y_model_raw, dtype=float)

    if delay_samples <= 0:
        return y_model

    shifted = np.zeros_like(y_model)
    if delay_samples < len(y_model):
        shifted[delay_samples:] = y_model[:-delay_samples]
    return shifted


def infer_delay_samples(model_params: dict[str, float], dt: float) -> int:
    if "delay_samples" in model_params:
        return max(0, int(round(float(model_params["delay_samples"]))))
    if "delay" in model_params:
        delay_seconds = float(model_params["delay"])
        if dt <= 0.0:
            return 0
        return max(0, int(round(delay_seconds / dt)))
    return 0


def calc_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    e = y_true - y_pred
    rmse = float(np.sqrt(np.mean(e**2)))
    mae = float(np.mean(np.abs(e)))
    y_span = float(np.max(y_true) - np.min(y_true))
    nrmse = float(rmse / y_span) if y_span > 1e-12 else float("nan")

    denom = float(np.linalg.norm(y_true - np.mean(y_true)))
    fit_percent = (
        float(100.0 * (1.0 - np.linalg.norm(e) / denom)) if denom > 1e-12 else float("nan")
    )

    return {
        "rmse": rmse,
        "mae": mae,
        "nrmse": nrmse,
        "fit_percent": fit_percent,
        "max_abs_error": float(np.max(np.abs(e))),
    }


def save_validation_plot(
    out_plot: Path,
    t: np.ndarray,
    u: np.ndarray,
    y_meas: np.ndarray,
    y_model: np.ndarray,
    err: np.ndarray,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), constrained_layout=True)

    axes[0].plot(t, u, color="tab:blue", linewidth=1.2, label="control u(t)")
    axes[0].set_title("Validation Input")
    axes[0].set_xlabel("Time [s]")
    axes[0].set_ylabel("u")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best")

    axes[1].plot(t, y_meas, color="tab:red", linewidth=1.2, label="measured y(t)")
    axes[1].plot(
        t,
        y_model,
        color="tab:green",
        linewidth=1.2,
        linestyle="--",
        label="model y_model(t)",
    )
    axes[1].set_title("Validation Output: measured vs model")
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("y")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best")

    axes[2].plot(t, err, color="tab:orange", linewidth=1.2, label="error e(t)=y_meas-y_model")
    axes[2].axhline(0.0, color="black", linewidth=0.8, alpha=0.7)
    axes[2].set_title("Validation Error")
    axes[2].set_xlabel("Time [s]")
    axes[2].set_ylabel("error")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc="best")

    out_plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_plot, dpi=140)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate identified model against a new LVM input/output pair"
    )
    parser.add_argument(
        "--model-json", type=Path, required=True, help="Model JSON from identification"
    )
    parser.add_argument("--control", type=Path, required=True, help="Validation control LVM file")
    parser.add_argument("--output", type=Path, required=True, help="Validation output LVM file")
    parser.add_argument(
        "--out-json", type=Path, required=True, help="Where to write validation metrics"
    )
    parser.add_argument(
        "--out-plot", type=Path, required=True, help="Where to write validation plot"
    )
    parser.add_argument(
        "--time-scale",
        type=float,
        default=0.1,
        help="Scale factor to convert raw LVM time into seconds (0.1 for deciseconds)",
    )
    args = parser.parse_args()

    if not args.model_json.exists():
        print(f"Model JSON not found: {args.model_json}")
        return 1
    if not args.control.exists() or not args.output.exists():
        print(f"Validation files missing. control={args.control} output={args.output}")
        return 1

    model_data = json.loads(args.model_json.read_text(encoding="utf-8"))
    model = model_data.get("model")
    if not isinstance(model, dict):
        print("Model JSON has no 'model' object.")
        return 2

    gz = model.get("discrete_z_domain", {})
    gz_num = np.asarray(gz.get("numerator", []), dtype=float)
    gz_den = np.asarray(gz.get("denominator", []), dtype=float)
    if gz_num.size == 0 or gz_den.size == 0:
        print("Discrete model coefficients missing in model JSON.")
        return 2

    params = model.get("parameters", {})
    if not isinstance(params, dict):
        params = {}

    t_u, u = read_lvm(args.control)
    t_y, y = read_lvm(args.output)

    t_u = np.asarray(t_u, dtype=float) * float(args.time_scale)
    t_y = np.asarray(t_y, dtype=float) * float(args.time_scale)
    t, u, y, dt = align_timebase(t_u, u, t_y, y)

    delay_samples = infer_delay_samples(params, dt)
    y_model = simulate_discrete(u, gz_num, gz_den, delay_samples)
    err = y - y_model

    metrics = calc_metrics(y, y_model)

    payload = {
        "validation_input": {
            "control_file": str(args.control),
            "output_file": str(args.output),
            "time_scale_to_seconds": float(args.time_scale),
            "sample_time_s": dt,
            "n_samples": int(len(t)),
        },
        "model_source": str(args.model_json),
        "model_used": {
            "name": model.get("name"),
            "family": model.get("family"),
            "order": model.get("order"),
            "discrete_z_domain": {
                "numerator": gz_num.tolist(),
                "denominator": gz_den.tolist(),
            },
            "applied_delay_samples": int(delay_samples),
        },
        "metrics": metrics,
        "signal_stats": {
            "u_min": float(np.min(u)),
            "u_max": float(np.max(u)),
            "y_meas_min": float(np.min(y)),
            "y_meas_max": float(np.max(y)),
            "y_model_min": float(np.min(y_model)),
            "y_model_max": float(np.max(y_model)),
        },
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    save_validation_plot(args.out_plot, t, u, y, y_model, err)

    print("Validation completed")
    print(f"- model_json: {args.model_json}")
    print(f"- out_json: {args.out_json}")
    print(f"- out_plot: {args.out_plot}")
    print(f"- rmse: {metrics['rmse']:.6f}")
    print(f"- fit_percent: {metrics['fit_percent']:.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
