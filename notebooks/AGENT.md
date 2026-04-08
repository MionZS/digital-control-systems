# AGENT.md — `notebooks/`

> For global context see `AGENT.md` at the repo root.

## What lives here

Interactive notebooks written in **[Marimo](https://marimo.io)** format.
Notebooks are plain `.py` files — version-controllable, diffable, and linted by ruff.

**Never create `.ipynb` files in this directory.**

| File | Description |
|---|---|
| `01_python_control_basics.py` | LTI models, step response, Bode plot, discretisation comparison |
| `02_sindy_from_sim.py` | Simulate MSD with LQR, then identify dynamics with SINDy |
| `03_closed_loop_compare_backends.py` | PID vs LQR closed-loop comparison with metrics table |
| `04_impulse_response_analysis.py` | Identify second-order parameters (K, zeta, omega_n, delay) from response data |
| `05_zoh_two_file_identification.py` | Read separate control/output files and identify G(z) and G(s) under ZOH |

## Running notebooks

```bash
# Open a single notebook in the browser
uv run marimo edit notebooks/01_python_control_basics.py

# Open a directory browser
uv run marimo edit notebooks/

# Run non-interactively (for CI / scripts)
uv run marimo run notebooks/01_python_control_basics.py
```

## Marimo notebook structure

Every notebook is a Python module with an `app = marimo.App()` and cells defined as
`@app.cell` decorated functions. Dependencies between cells are resolved automatically
via function arguments.

```python
import marimo
app = marimo.App(width="medium")

@app.cell
def _():
    import numpy as np
    return (np,)          # ← must return every name used by other cells

@app.cell
def _(np):                # ← receives np from the cell above
    x = np.linspace(0, 1, 100)
    return (x,)

if __name__ == "__main__":
    app.run()
```

Key rules:
- **Return every name** that downstream cells need (as a tuple).
- **Receive names** you depend on as function arguments.
- `mo.md("...")` renders markdown. Use `mo.md(f"...")` for dynamic content.
- Return a `matplotlib.figure.Figure` from a cell to render it inline.
- Call `matplotlib.use("Agg")` near the top to avoid GUI conflicts.

## Conventions for this repo's notebooks

- Notebooks **orchestrate** calls to `src/control_lab` — no algorithm logic in notebooks.
- Set the seed with `set_global_seed(42)` at the start of any notebook that uses
  random data or SINDy identification.
- Close figures with `plt.close("all")` after returning the `Figure` object to avoid
  memory leaks in long sessions.
- When `pysindy` or other optional deps are absent, show a `mo.md("⚠️ ...")` message
  and set the identifier variable to `None` so downstream cells can guard on it.

## When adding a new notebook

1. Name it `NN_<short_description>.py` (two-digit prefix for ordering).
2. Start with a `mo.md(...)` cell that explains the notebook purpose.
3. Keep all imports in a dedicated cell near the top.
4. Add the notebook to the table at the top of this file.
5. Update `.github/copilot-instructions.md` if the notebook demonstrates a new
   capability or design pattern.

## References

- Marimo docs: https://docs.marimo.io
- Marimo cell API: https://docs.marimo.io/api/cells/
- Marimo UI elements: https://docs.marimo.io/api/inputs/
