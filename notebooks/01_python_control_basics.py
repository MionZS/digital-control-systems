"""Notebook 01 — python-control basics.

Demonstrates creating LTI models, computing step responses, and Bode plots
using the control-lab framework.

Run with:  uv run marimo edit notebooks/01_python_control_basics.py
"""

import marimo

__generated_with = "0.20.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # 01 — python-control basics

    This notebook demonstrates:
    - Creating an LTI model (mass-spring-damper)
    - Step response simulation
    - Bode plot and stability margins
    - Poles and zeros
    """)
    return


@app.cell
def _():
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
    matplotlib.use("Agg")  # non-interactive backend for marimo

    from control_lab.design.pid import PIDController
    from control_lab.models.lti import LTIModel
    from control_lab.sim.backend_control import ControlBackend
    from control_lab.sim.common import compute_metrics
    from control_lab.utils.plotting import plot_bode, plot_step_response

    return (
        ControlBackend,
        LTIModel,
        PIDController,
        compute_metrics,
        np,
        plot_bode,
        plot_step_response,
        plt,
    )


@app.cell
def _(LTIModel, mo):
    # --- Create a mass-spring-damper model ---
    # Parameters:  m=1 kg,  k=1 N/m,  c=0.5 N·s/m
    model = LTIModel.mass_spring_damper(m=1.0, k=1.0, c=0.5)

    mo.md(f"""
    ## Mass-Spring-Damper model (continuous-time)

    | Property | Value |
    |---|---|
    | States | {model.n_states} (position, velocity) |
    | Inputs | {model.n_inputs} (force) |
    | Outputs | {model.n_outputs} (position) |
    | Stable  | {model.is_stable()} |
    | Poles   | {model.poles().round(4).tolist()} |
    """)
    return


@app.cell
def _(
    ControlBackend,
    LTIModel,
    PIDController,
    compute_metrics,
    np,
    plot_step_response,
    plt,
):
    # --- Closed-loop step response with PID ---
    model_sim = LTIModel.mass_spring_damper(m=1.0, k=1.0, c=0.5)
    backend = ControlBackend()

    pid = PIDController(kp=20.0, ki=5.0, kd=3.0, dt=0.01, u_min=-100.0, u_max=100.0)

    def r_step(t):
        return np.array([1.0])

    result = backend.simulate(model_sim, pid, np.array([0.0, 0.0]), r_step, (0.0, 10.0), 0.01)
    metrics = compute_metrics(result, r_final=1.0)

    fig = plot_step_response(result, r_final=1.0, title="MSD + PID — step response")
    plt.close("all")

    print("Metrics:", {k: round(v, 4) for k, v in metrics.items()})
    fig
    return


@app.cell
def _(LTIModel, plot_bode, plt):
    # --- Bode plot of the open-loop plant ---
    model_bode = LTIModel.mass_spring_damper(m=1.0, k=1.0, c=0.5)
    fig_bode = plot_bode(model_bode, title="Open-loop Bode — Mass-Spring-Damper")
    plt.close("all")
    fig_bode
    return


@app.cell
def _(LTIModel, mo):
    # --- Discretisation comparison ---
    from control_lab.design.discretization import compare_methods

    model_d = LTIModel.mass_spring_damper(m=1.0, k=1.0, c=0.5)
    disc_results = compare_methods(model_d.A, model_d.B, dt=0.05)

    rows = []
    for method, data in disc_results.items():
        if "error" in data:
            rows.append(f"| {method} | ❌ {data['error']} |")
        else:
            eigs = [round(e, 4) for e in sorted(abs(v) for v in __import__("numpy").linalg.eigvals(data["Ad"]))]
            rows.append(f"| {method} | ✅ {eigs} |")

    mo.md(
        "## Discretisation methods (dt=0.05 s)\n\n"
        "| Method | |Ad| eigenvalues |\n|---|---|\n"
        + "\n".join(rows)
    )
    return


if __name__ == "__main__":
    app.run()
