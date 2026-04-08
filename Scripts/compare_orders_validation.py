"""Fit first/second/third-order candidates on EstimacaoBox4 and validate on ValidacaoBox4.

Outputs:
- output/box4/all_orders_comparison.json
- output/box4/all_orders_comparison.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# make Scripts importable
sys.path.insert(0, str(Path(__file__).resolve().parent))
import automatic_order_identification as aio

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = Path("output/box4")
OUT_DIR.mkdir(parents=True, exist_ok=True)

EST_CTRL = Path("input/EstimacaoBox4/controle.lvm")
EST_OUT = Path("input/EstimacaoBox4/saida.lvm")
VAL_CTRL = Path("input/ValidacaoBox4/controle.lvm")
VAL_OUT = Path("input/ValidacaoBox4/saida.lvm")

NAMES = ["first_order", "second_order_underdamped", "third_order_real_poles"]


def simple_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
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


def main() -> int:
    # load estimation data and build regions
    t_e, u_e = aio.read_lvm(EST_CTRL)
    t_y_e, y_e = aio.read_lvm(EST_OUT)
    t_e = np.asarray(t_e, dtype=float) * 0.1
    t_y_e = np.asarray(t_y_e, dtype=float) * 0.1
    t_e, u_e, y_e, dt_e = aio.align_timebase(t_e, u_e, t_y_e, y_e)

    step_idx_e = aio.detect_steps(u_e, threshold=0.5)
    regions = aio.build_regions(
        t=t_e,
        u=u_e,
        y=y_e,
        step_idx=step_idx_e,
        pre_samples=15,
        post_ignore=3,
        steady_tail=20,
        fit_horizon_s=80.0,
        min_region_len=40,
    )
    regions_norm = aio.normalized_regions(regions, fit_horizon_s=80.0)

    candidates = [aio.fit_candidate(name, regions_norm, dt_e) for name in NAMES]

    # load validation set
    t_v, u_v = aio.read_lvm(VAL_CTRL)
    t_y_v, y_v = aio.read_lvm(VAL_OUT)
    t_v = np.asarray(t_v, dtype=float) * 0.1
    t_y_v = np.asarray(t_y_v, dtype=float) * 0.1
    t_v, u_v, y_v, dt_v = aio.align_timebase(t_v, u_v, t_y_v, y_v)
    step_idx_v = aio.detect_steps(u_v, threshold=0.5)

    results = []

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), constrained_layout=True)

    for ax, cand in zip(axes, candidates):
        # simulate candidate on validation input
        y_sim = aio.simulate_output_from_input(cand, t_v, u_v, y_v, step_idx_v)
        if y_sim is None:
            y_sim = np.zeros_like(y_v)
        metrics = simple_metrics(y_v, y_sim)
        results.append(
            {
                "name": cand.name,
                "order": cand.order,
                "params": cand.params,
                "metrics": metrics,
            }
        )

        ax.plot(t_v, u_v, color="tab:blue", linewidth=1.0, label="control u(t)")
        ax.plot(t_v, y_v, color="tab:red", linewidth=1.0, label="measured y(t)")
        ax.plot(
            t_v, y_sim, color="tab:green", linestyle="--", linewidth=1.0, label=f"model {cand.name}"
        )
        ax.set_title(
            f"{cand.name} (order {cand.order}) — RMSE {metrics['rmse']:.3f} fit% {metrics['fit_percent']:.1f}"
        )
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

    OUT_JSON = OUT_DIR / "all_orders_comparison.json"
    OUT_PNG = OUT_DIR / "all_orders_comparison.png"
    OUT_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")
    fig.savefig(OUT_PNG, dpi=140)
    plt.close(fig)

    print("Wrote:", OUT_JSON, OUT_PNG)
    for r in results:
        print(
            f"{r['name']}: rmse={r['metrics']['rmse']:.3f} fit%={r['metrics']['fit_percent']:.2f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
