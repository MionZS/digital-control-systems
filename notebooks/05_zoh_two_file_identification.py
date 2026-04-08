"""Notebook 05 - Two-file ZOH identification for G(z) and G(s).

Run with: uv run marimo edit notebooks/05_zoh_two_file_identification.py
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
    # 05 - ZOH identification from two files (control and output)

    This notebook reads two files:
    - control signal `u(t)`
    - output signal `y(t)`

    It estimates:
    - continuous model `G(s)` from step-response characteristics
        - discrete model `G(z)` with exact ZOH (equivalent to the lecture procedure
            based on sampled step response)
    """)
    return


@app.cell
def _():
    import json
    from pathlib import Path

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from control_lab.ident.zoh_ident import identify_zoh_from_second_order, tf_string_s, tf_string_z

    return (
        Path,
        identify_zoh_from_second_order,
        json,
        plt,
        tf_string_s,
        tf_string_z,
    )


@app.cell
def _(Path, json):
    defaults_path = Path("input/impulse_response/zoh_defaults.json")
    fallback_control = Path("input/degrausMotorTQ/controle.lvm")
    fallback_output = Path("input/degrausMotorTQ/saida.lvm")

    defaults = {
        "control_file": str(fallback_control),
        "output_file": str(fallback_output),
        "output_model_json": "input/impulse_response/zoh_identified_model.json",
    }

    if defaults_path.exists():
        with defaults_path.open("r", encoding="utf-8") as fp:
            defaults.update(json.load(fp))

    control_file = Path(defaults["control_file"])
    output_file = Path(defaults["output_file"])
    output_json = Path(defaults["output_model_json"])
    return control_file, defaults, output_file, output_json


@app.cell
def _(defaults, mo):
    mo.md(
        "## Effective defaults\n\n"
        + "\n".join(
            [
                f"- `control_file`: `{defaults['control_file']}`",
                f"- `output_file`: `{defaults['output_file']}`",
                f"- `output_model_json`: `{defaults['output_model_json']}`",
            ]
        )
    )
    return


@app.cell
def _(
    control_file,
    identify_zoh_from_second_order,
    mo,
    output_file,
    tf_string_s,
    tf_string_z,
):
    result = None
    t = None
    u = None
    y = None

    if not control_file.exists() or not output_file.exists():
        mo.md(
            "Input files not found. Provide both files and rerun:\n"
            f"- control: `{control_file}`\n"
            f"- output: `{output_file}`"
        )
    else:
        result, t, u, y = identify_zoh_from_second_order(
            control_path=control_file,
            output_path=output_file,
        )
        gz = tf_string_z(result.num_z, result.den_z)
        gs = tf_string_s(result.num_s, result.den_s)

        mo.md(
            "## Identified models\n\n"
            f"- Sample time `dt`: `{result.dt:.6f}` s\n"
            f"- Method: `{result.method}`\n"
            f"- Fit RMSE: `{result.fit_rmse:.6f}`\n"
            f"- Estimated `K`: `{result.gain:.6f}`\n"
            f"- Estimated `ζ`: `{result.zeta:.6f}`\n"
            f"- Estimated `ω_n`: `{result.omega_n:.6f}` rad/s\n"
            f"- Estimated `delay`: `{result.delay:.6f}` s\n"
            f"- `{gz}`\n"
            f"- `{gs}`"
        )
    return result, t, u, y


@app.cell
def _(json, output_json, result):
    if result is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "dt": float(result.dt),
            "method": result.method,
            "gain": float(result.gain) if result.gain is not None else None,
            "zeta": float(result.zeta) if result.zeta is not None else None,
            "omega_n": float(result.omega_n) if result.omega_n is not None else None,
            "delay": float(result.delay) if result.delay is not None else None,
            "num_z": [float(v) for v in result.num_z],
            "den_z": [float(v) for v in result.den_z],
            "num_s": [float(v) for v in result.num_s],
            "den_s": [float(v) for v in result.den_s],
            "fit_rmse": float(result.fit_rmse),
        }
        output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return


@app.cell
def _(plt, result, t, u, y):
    fig = None
    if result is None or t is None or u is None or y is None:
        pass
    else:
        fig, axes = plt.subplots(3, 1, figsize=(9, 8), constrained_layout=True)

        axes[0].plot(t, u, linewidth=1.2)
        axes[0].set_title("Control input u(t)")
        axes[0].set_xlabel("Time [s]")
        axes[0].set_ylabel("u")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(t, y, linewidth=1.2, label="measured y")
        axes[1].plot(t, result.y_hat, linewidth=1.2, linestyle="--", label="model fit")
        axes[1].set_title("Output y(t) and model fit")
        axes[1].set_xlabel("Time [s]")
        axes[1].set_ylabel("y")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(t, y - result.y_hat, linewidth=1.2)
        axes[2].set_title("Residual y - y_hat")
        axes[2].set_xlabel("Time [s]")
        axes[2].set_ylabel("error")
        axes[2].grid(True, alpha=0.3)

    return (fig,)


@app.cell
def _(fig):
    fig
    return


if __name__ == "__main__":
    app.run()
