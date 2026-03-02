# AGENT.md — `src/control_lab/utils/`

> For package-level context see `src/control_lab/AGENT.md`.

## What lives here

Cross-cutting utilities with **no internal `control_lab` dependencies**.
These can be imported from any layer without creating circular imports.

| File | Purpose |
|---|---|
| `logging.py` | `get_logger(name)` — Rich-based logger |
| `plotting.py` | Matplotlib helpers for step responses, Bode plots, trajectory overlays |
| `seeding.py` | `set_global_seed(seed)` — numpy + stdlib random |

## `logging.py` — key facts

```python
from control_lab.utils.logging import get_logger
logger = get_logger(__name__)
logger.info("Metrics: %s", metrics)
```

- Uses `rich.logging.RichHandler` for colour output with tracebacks.
- `get_logger` is idempotent — safe to call multiple times with the same name.
- Do **not** call `logging.basicConfig()` anywhere in the library code; leave that to
  the user's application / experiment script.

## `plotting.py` — key facts

All functions use **lazy imports** (`import matplotlib.pyplot as plt` inside the function
body) so the module is importable in headless environments.

| Function | Inputs | Returns |
|---|---|---|
| `plot_step_response(result, r_final, title, save_path)` | `SimulationResult` | `Figure` |
| `plot_bode(model, title, save_path)` | `LTIModel` | `Figure` |
| `plot_trajectories(results_dict, title, save_path)` | `dict[str, SimulationResult]` | `Figure` |

- `save_path` is optional; if provided, the figure is saved with `dpi=150`.
- `results_dict` maps label strings to `SimulationResult` objects — used for overlaying
  multiple controllers on the same axes.
- `plot_bode` uses `python-control`'s `control.bode_plot`; always consult
  https://python-control.readthedocs.io/en/latest/generated/control.bode_plot.html
  for the current API.

### In Marimo notebooks
Use `matplotlib.use("Agg")` at the top of the notebook to avoid display conflicts,
then return the `Figure` from the cell so Marimo renders it inline.

## `seeding.py` — key facts

```python
from control_lab.utils.seeding import set_global_seed
set_global_seed(42)  # sets numpy.random.seed and random.seed
```

Call this at the start of every experiment script and at the top of test functions
that involve random data generation.

## When adding a new utility

- Keep it stateless and side-effect-free where possible.
- Use lazy imports for heavy dependencies (matplotlib, control, etc.).
- Do not import from other `control_lab` sub-packages.
- Add a test if the utility has non-trivial logic.
