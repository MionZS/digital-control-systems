"""Notebook 04 - Identify a second-order response model from input data.

Run with: uv run marimo edit notebooks/04_impulse_response_analysis.py
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
    # 04 - Second-order model extraction

    This notebook loads response data from `input/impulse_response/`,
    estimates a second-order transfer function, and exports the identified model to JSON.
    """)
    return


@app.cell
def _():
    import json
    from pathlib import Path

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from control_lab.ident.second_order import (
        estimate_second_order_step_model,
        load_step_response_data,
        second_order_summary,
    )

    return (
        Path,
        json,
        estimate_second_order_step_model,
        load_step_response_data,
        second_order_summary,
        plt,
    )


@app.cell
def _(Path, json):
    defaults_path = Path("input/impulse_response/notebook_defaults.json")
    fallback_input = Path("input/impulse_response/sample_thermal_step.txt")

    defaults = {
        "input_csv": str(fallback_input),
        "output_model_json": "input/impulse_response/identified_model.json",
    }

    if defaults_path.exists():
        with defaults_path.open("r", encoding="utf-8") as fp:
            loaded = json.load(fp)
        defaults.update(loaded)

    input_csv = Path(defaults["input_csv"])
    model_out = Path(defaults["output_model_json"])
    return defaults, input_csv, model_out


@app.cell
def _(defaults, mo):
    mo.md(
        "## Effective defaults\n\n"
        + "\n".join(
            [
                f"- `input_csv`: `{defaults['input_csv']}`",
                f"- `output_model_json`: `{defaults['output_model_json']}`",
            ]
        )
    )
    return


@app.cell
def _(
    input_csv,
    estimate_second_order_step_model,
    load_step_response_data,
    second_order_summary,
    mo,
):
    if not input_csv.exists():
        mo.md(
            f"Input file not found: `{input_csv}`. Add a `.csv` (columns `t,y`) or `.txt` "
            "step-response file to `input/impulse_response/` and restart the notebook."
        )
        model = None
        summary = None
        t = None
        y = None
    else:
        data = load_step_response_data(input_csv)
        model = estimate_second_order_step_model(data)
        summary = second_order_summary(model)
        t = data.t
        y = data.y

        mo.md(
            "## Second-order identification summary\n\n"
            + "\n".join(
                [
                    f"- Gain `K`: `{summary['gain']:.6f}`",
                    f"- Damping ratio `ζ`: `{summary['zeta']:.6f}`",
                    f"- Natural frequency `ω_n`: `{summary['omega_n']:.6f}` rad/s",
                    f"- Delay `θ`: `{summary['delay']:.6f}` s",
                    f"- Overshoot: `{summary['overshoot']:.6f}`",
                    f"- Rise time: `{summary['rise_time']:.6f}` s",
                    f"- Settling time: `{summary['settling_time']:.6f}` s",
                ]
            )
        )
    return model, t, y


@app.cell
def _(json, model, model_out):
    if model is not None:
        model_out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "gain": float(model.gain),
            "zeta": float(model.zeta),
            "omega_n": float(model.omega_n),
            "delay": float(model.delay),
            "polynomial": model.polynomial,
            "numerator": [float(v) for v in model.numerator],
            "denominator": [float(v) for v in model.denominator],
        }
        model_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return


@app.cell
def _(model, plt, t, y):
    if model is None or t is None or y is None:
        return

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), constrained_layout=True)

    axes[0].plot(t, y, linewidth=1.2)
    axes[0].axhline(model.y_ss, color="gray", linestyle="--", linewidth=1.0)
    axes[0].set_title("Measured response")
    axes[0].set_xlabel("Time [s]")
    axes[0].set_ylabel("y(t)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, y - y[0], linewidth=1.5)
    axes[1].axhline(model.y_ss - model.y0, color="gray", linestyle="--", linewidth=1.0)
    axes[1].set_title("Response relative to initial value")
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Δy")
    axes[1].grid(True, alpha=0.3)

    plt.close("all")

    return fig


if __name__ == "__main__":
    app.run()
