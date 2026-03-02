"""Notebook 02 — SINDy identification from simulation data.

Simulate the mass-spring-damper with an LQR controller, collect the
state/input trajectory, then identify the dynamics with SINDy and validate
the identified model.

Run with:  uv run marimo edit notebooks/02_sindy_from_sim.py
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
        # 02 — SINDy identification from simulation data

        Pipeline:
        1. Simulate the mass-spring-damper under LQR control (rich excitation)
        2. Collect state / input trajectory as the identification dataset
        3. Fit a SINDy model (polynomial feature library)
        4. Validate with one-step prediction and full rollout
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
    from control_lab.models.lti import LTIModel
    from control_lab.sim.backend_control import ControlBackend
    from control_lab.utils.seeding import set_global_seed

    set_global_seed(42)

    return (
        ControlBackend,
        LQRController,
        LTIModel,
        lqr_continuous,
        matplotlib,
        np,
        plt,
        set_global_seed,
    )


@app.cell
def _(ControlBackend, LQRController, LTIModel, lqr_continuous, mo, np):
    # --- Generate rich identification data ---
    model = LTIModel.mass_spring_damper(m=1.0, k=1.0, c=0.5)

    Q = np.diag([10.0, 1.0])
    R = np.eye(1) * 0.1
    K, _, _ = lqr_continuous(model.A, model.B, Q, R)
    ctrl = LQRController(K)

    backend = ControlBackend()

    def r_chirp(t):
        return np.array([np.sin(2 * np.pi * 0.3 * t), 0.0])

    result = backend.simulate(model, ctrl, np.array([0.5, 0.0]), r_chirp, (0.0, 30.0), 0.01)

    data = {"x": result.x, "u": result.u, "t": result.t}

    mo.md(f"""
    ## Identification dataset

    | | Value |
    |---|---|
    | Samples | {result.x.shape[0]} |
    | States  | {result.x.shape[1]} (position, velocity) |
    | Inputs  | {result.u.shape[1]} (force) |
    | Duration | {result.t[-1]:.1f} s |
    """)
    return (data,)


@app.cell
def _(data, mo):
    try:
        from control_lab.ident.feature_library import PolynomialLibrary
        from control_lab.ident.sindy_fit import SINDyIdentifier

        lib = PolynomialLibrary(degree=2)
        identifier = SINDyIdentifier(feature_library=lib, dt=0.01)
        identifier.fit(data)
        equations = identifier.get_equations()

        mo.md(
            "## Discovered equations\n\n"
            + "\n\n".join(f"- `ẋ{i} = {eq}`" for i, eq in enumerate(equations))
        )
    except ImportError:
        mo.md("⚠️  **pysindy not installed** — run `uv add pysindy` to enable this cell.")
        identifier = None
        equations = []
    return (identifier,)


@app.cell
def _(data, identifier, mo, np, plt):
    if identifier is None:
        mo.md("*SINDy not available — skipping validation.*")
    else:
        from control_lab.ident.sindy_validate import plot_validation, rollout_error

        rmse = rollout_error(identifier, data)

        # Split: fit on first 2/3, validate on last 1/3
        n = len(data["t"])
        split = 2 * n // 3
        val_data = {
            "x": data["x"][split:],
            "u": data["u"][split:],
            "t": data["t"][split:],
        }
        _fig = plot_validation(identifier, val_data, title=f"SINDy validation (RMSE={rmse:.4f})")
        plt.close("all")

        mo.md(f"**Rollout RMSE:** `{rmse:.6f}`")


if __name__ == "__main__":
    app.run()
