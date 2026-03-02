"""Notebook 03 — Closed-loop comparison: PID vs LQR.

Simulate the mass-spring-damper under both PID and LQR control and compare
step-response metrics side-by-side.

Run with:  uv run marimo edit notebooks/03_closed_loop_compare_backends.py
"""

import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(
        r"""
        # 03 — Closed-loop comparison: PID vs LQR

        Compare two controllers on the same mass-spring-damper plant:
        - **PID** with hand-tuned gains and anti-windup
        - **LQR** designed from a quadratic cost (optimal state feedback)

        Metrics: overshoot, settling time, IAE, ISE, control effort (∫u²).
        """
    )
    return


@app.cell
def _():
    import matplotlib
    import numpy as np
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from control_lab.design.lqr import LQRController, lqr_continuous
    from control_lab.design.pid import PIDController
    from control_lab.models.lti import LTIModel
    from control_lab.sim.backend_control import ControlBackend
    from control_lab.sim.common import compute_metrics
    from control_lab.utils.plotting import plot_trajectories

    return (
        ControlBackend,
        LQRController,
        LTIModel,
        PIDController,
        compute_metrics,
        lqr_continuous,
        matplotlib,
        np,
        plt,
        plot_trajectories,
    )


@app.cell
def _(ControlBackend, LQRController, LTIModel, PIDController, compute_metrics, lqr_continuous, np):
    model = LTIModel.mass_spring_damper(m=1.0, k=1.0, c=0.5)
    backend = ControlBackend()
    x0 = np.array([0.0, 0.0])

    def r_step(t):
        return np.array([1.0])

    def r_step_full(t):
        return np.array([1.0, 0.0])

    # PID
    pid = PIDController(kp=20.0, ki=5.0, kd=3.0, dt=0.01, u_min=-100.0, u_max=100.0)
    result_pid = backend.simulate(model, pid, x0, r_step, (0.0, 10.0), 0.01)
    metrics_pid = compute_metrics(result_pid, r_final=1.0)

    # LQR
    Q = np.diag([100.0, 1.0])  # penalise position error heavily
    R = np.eye(1) * 0.1
    K, _, _ = lqr_continuous(model.A, model.B, Q, R)
    lqr_ctrl = LQRController(K)
    result_lqr = backend.simulate(model, lqr_ctrl, x0, r_step_full, (0.0, 10.0), 0.01)
    metrics_lqr = compute_metrics(result_lqr, r_final=1.0)

    return (
        K,
        Q,
        R,
        backend,
        lqr_ctrl,
        metrics_lqr,
        metrics_pid,
        model,
        pid,
        r_step,
        r_step_full,
        result_lqr,
        result_pid,
        x0,
    )


@app.cell
def _(metrics_lqr, metrics_pid, mo):
    def _fmt(v):
        return f"{v:.4f}" if isinstance(v, float) and v != float("inf") else str(v)

    rows = "\n".join(
        f"| {k} | {_fmt(metrics_pid[k])} | {_fmt(metrics_lqr[k])} |"
        for k in metrics_pid
    )
    mo.md(
        "## Metrics comparison\n\n"
        "| Metric | PID | LQR |\n|---|---|---|\n"
        + rows
    )


@app.cell
def _(mo, plt, plot_trajectories, result_lqr, result_pid):
    _fig_traj = plot_trajectories(
        {"PID": result_pid, "LQR": result_lqr},
        title="Step response comparison — PID vs LQR",
    )
    plt.close("all")
    mo.md("## Output trajectories")


@app.cell
def _(mo, np, plt, result_lqr, result_pid):
    # Control effort comparison
    fig2, ax = plt.subplots(figsize=(10, 4))
    u_pid = result_pid.u[:, 0]
    u_lqr = result_lqr.u[:, 0]
    ax.plot(result_pid.t, u_pid, label="PID", linewidth=2)
    ax.plot(result_lqr.t, u_lqr, "--", label="LQR", linewidth=2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Control input (N)")
    ax.set_title("Control effort")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    effort_pid = float(np.trapezoid(u_pid**2, result_pid.t))
    effort_lqr = float(np.trapezoid(u_lqr**2, result_lqr.t))
    plt.close("all")
    mo.md(f"**∫u² — PID:** `{effort_pid:.2f}` | **∫u² — LQR:** `{effort_lqr:.2f}`")


if __name__ == "__main__":
    app.run()
