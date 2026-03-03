"""CLI entry point for running a control experiment from a YAML config file."""

from __future__ import annotations

import argparse
import csv
import datetime
import json
from pathlib import Path

import numpy as np
import yaml


def main(config_path: str, backend_name: str = "control") -> None:
    """Load *config_path*, run the experiment, and save artefacts."""
    with open(config_path) as fh:
        config = yaml.safe_load(fh)

    name = config.get("name", "experiment")
    seed = config.get("seed", 0)

    from control_lab.utils.seeding import set_global_seed

    set_global_seed(seed)

    # ------------------------------------------------------------------ model
    model_cfg = config["model"]
    if model_cfg["type"] == "mass_spring_damper":
        from control_lab.models.lti import LTIModel

        model = LTIModel.mass_spring_damper(
            m=float(model_cfg["m"]),
            k=float(model_cfg["k"]),
            c=float(model_cfg["c"]),
            dt=model_cfg.get("dt"),
        )
    else:
        raise ValueError(f"Unknown model type: {model_cfg['type']!r}")

    # -------------------------------------------------------------- controller
    ctrl_cfg = config["controller"]
    if ctrl_cfg["type"] == "pid":
        from control_lab.design.pid import PIDController

        controller = PIDController(
            kp=float(ctrl_cfg["kp"]),
            ki=float(ctrl_cfg["ki"]),
            kd=float(ctrl_cfg["kd"]),
            dt=float(ctrl_cfg["dt"]),
            u_min=float(ctrl_cfg.get("u_min", float("-inf"))),
            u_max=float(ctrl_cfg.get("u_max", float("inf"))),
        )
    elif ctrl_cfg["type"] == "lqr":
        from control_lab.design.lqr import LQRController, lqr_continuous

        Q = np.array(ctrl_cfg["Q"])
        R = np.array(ctrl_cfg["R"])
        K, _, _ = lqr_continuous(model.A, model.B, Q, R)
        controller = LQRController(K)
    else:
        raise ValueError(f"Unknown controller type: {ctrl_cfg['type']!r}")

    # --------------------------------------------------------------- simulate
    sim_cfg = config["simulation"]
    x0 = np.array(sim_cfg["x0"], dtype=float)
    r_val = float(sim_cfg["r"])
    t_span = (float(sim_cfg["t_span"][0]), float(sim_cfg["t_span"][1]))
    dt = float(sim_cfg["dt"])

    def r_func(t: float) -> np.ndarray:  # noqa: ARG001
        return np.array([r_val])

    from control_lab.sim.common import compute_metrics

    if backend_name == "control":
        from control_lab.sim.backend_control import ControlBackend

        backend = ControlBackend()
    elif backend_name == "collimator":
        try:
            from control_lab.sim.backend_collimator import CollimatorBackend

            backend = CollimatorBackend()
        except ImportError as exc:  # pragma: no cover - runtime optional dependency
            raise ImportError(
                "Collimator backend requested but optional dependency is missing: "
                "install with `uv add control-lab[collimator]` or use --backend control"
            ) from exc
    else:
        raise ValueError(f"Unknown backend: {backend_name!r}")
    result = backend.simulate(model, controller, x0, r_func, t_span, dt)
    metrics = compute_metrics(result, r_final=r_val)

    # ---------------------------------------------------------- save artefacts
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    run_dir = Path(f"experiments/results/{timestamp}__{name}__seed{seed}")
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2)

    with open(run_dir / "config_used.yaml", "w") as fh:
        yaml.dump(config, fh)

    with open(run_dir / "trajectory.csv", "w", newline="") as fh:
        writer = csv.writer(fh)
        x_cols = [f"x{i}" for i in range(result.x.shape[1])]
        y_cols = [f"y{i}" for i in range(result.y.shape[1])]
        u_cols = [f"u{i}" for i in range(result.u.shape[1])]
        writer.writerow(["t"] + x_cols + y_cols + u_cols)
        for idx in range(len(result.t)):
            row = (
                [float(result.t[idx])]
                + list(map(float, result.x[idx]))
                + list(map(float, result.y[idx]))
                + list(map(float, result.u[idx]))
            )
            writer.writerow(row)

    from control_lab.utils.logging import get_logger

    logger = get_logger(__name__)
    logger.info("Results saved to %s", run_dir)
    logger.info("Metrics: %s", metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a control-lab experiment.")
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    parser.add_argument(
        "--backend",
        choices=["control", "collimator"],
        default="control",
        help="Which simulation backend to use.",
    )
    args = parser.parse_args()
    main(args.config, backend_name=args.backend)
