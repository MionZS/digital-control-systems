# GitHub Copilot — Repository Instructions

## What this repository is

`control-lab` is a **modular Python framework** for digital control-systems research and education.
It bridges the gap between paper → simulation → real experiment by providing stable, composable
interfaces for modelling, controller design, simulation, and data-driven identification.

Target audience: control engineers and researchers who want a reproducible, scriptable
workflow that can grow from a classroom demo to a hardware bench test without rewriting
the core logic.

---

## Tooling and environment

| Tool | Role | Notes |
|---|---|---|
| **Python 3.12** | Primary runtime | Tested in CI; all type annotations use 3.12+ syntax |
| **uv** | Package manager & virtual-env | Use `uv add`, `uv run`, `uv sync`; never raw `pip` |
| **Marimo** | Interactive notebooks | `.py` files in `notebooks/`; run with `uv run marimo edit <file>` |
| **ruff** | Linter + formatter | Run `uv run ruff check .` / `uv run ruff format .` |
| **mypy** | Static type checking | Loose mode (`strict = false`); `ignore_missing_imports = true` |
| **pytest** | Test runner | `uv run pytest tests/ -v` |
| **hatchling** | Build backend | Configured in `pyproject.toml` |

Python 3.13 is tested in CI (optional). Python 3.14 gets smoke-test only (`continue-on-error`).

---

## Module map — where to find each type of code

```
src/control_lab/
│
├── interfaces.py          Protocol definitions (ModelProtocol, ControllerProtocol,
│                          SimulatorBackendProtocol, IdentifierProtocol)
│
├── models/
│   ├── lti.py             LTIModel dataclass — continuous/discrete state-space,
│   │                      ZOH discretisation, poles/zeros, mass_spring_damper() factory
│   ├── nonlinear.py       NonlinearModel dataclass — f(x,u,t) / g(x,u,t) callables,
│   │                      inverted_pendulum() factory
│   └── datasets.py        Signal generators: PRBS, multi-sine
│
├── design/
│   ├── pid.py             PIDController — discrete, anti-windup (clamping), derivative filter
│   ├── lqr.py             lqr_continuous/lqr_discrete functions + LQRController class
│   ├── observers.py       luenberger_gain, kalman_gain, StateObserver
│   └── discretization.py  discretize(A, B, dt, method) + compare_methods()
│
├── sim/
│   ├── common.py          SimulationResult dataclass + compute_metrics()
│   ├── backend_control.py ControlBackend — scipy RK45 for LTI/nonlinear, algebraic for discrete
│   └── backend_collimator.py  Optional pycollimator backend (graceful ImportError)
│
├── ident/
│   ├── feature_library.py  PolynomialLibrary / FourierLibrary thin wrappers (lazy pysindy import)
│   ├── sindy_fit.py        SINDyIdentifier — fit(data), predict(x0,u,t), get_equations()
│   └── sindy_validate.py   one_step_error, rollout_error, plot_validation
│
├── utils/
│   ├── logging.py         get_logger(name) — Rich-based logger
│   ├── plotting.py        plot_step_response, plot_bode, plot_trajectories
│   └── seeding.py         set_global_seed(seed) — numpy + stdlib random
│
└── experiments/
    ├── run_experiment.py  CLI entry point (--config path/to/config.yaml)
    └── configs/           YAML experiment configs (demo_a.yaml, ...)
```

Tests live in `tests/`, notebooks in `notebooks/`, CI in `.github/workflows/ci.yml`.

---

## Core design decisions

### 1. Protocol-based interfaces (`interfaces.py`)
All cross-layer dependencies go through `Protocol` classes — never concrete imports from other
layers. A new controller only needs to implement `compute(x, r, t)` and `reset()`. A new backend
only needs to implement `simulate(...)`. This keeps layers swappable without ripple changes.

### 2. `src` layout
All importable code lives under `src/control_lab/`. The project is installed as a package
(`uv sync`) so notebooks and tests import from `control_lab.*`, never from relative paths.

### 3. Lazy optional imports
`pysindy`, `control` (python-control), and `pycollimator` are imported *inside* the functions
that need them, never at module top level. This lets the package import cleanly even when
optional deps are absent. Pattern:
```python
def fit(self, data):
    import pysindy as ps  # lazy — only needed here
    ...
```

### 4. Marimo notebooks (not Jupyter)
Notebooks are `.py` files that use `marimo`'s `@app.cell` decorator. They are plain Python,
diffable, and linted by ruff. Never create `.ipynb` files. Run with:
```bash
uv run marimo edit notebooks/<name>.py
```

### 5. Reproducible experiments via YAML configs
Every experiment run is driven by a YAML config (seed, model params, controller, sim settings).
Results land in `experiments/results/` which is **gitignored**. Only configs are version-controlled.

### 6. Uppercase matrix names are intentional
Control theory conventionally uses uppercase for matrices (A, B, C, D, K, Q, R, P).
Ruff rules `N803`, `N806`, `N802`, `E741` are disabled in `pyproject.toml` to allow this.

---

## Key libraries — always refer to their documentation

When generating or reviewing code that uses these libraries, **always consult their current
documentation** rather than relying on potentially stale knowledge. Use idiomatic, documented APIs.

| Library | Docs URL | Notes |
|---|---|---|
| `python-control` | https://python-control.readthedocs.io | `control.ss`, `control.bode_plot`, `control.place`, `control.lqr` |
| `pysindy` | https://pysindy.readthedocs.io | `SINDy`, `PolynomialLibrary`, `STLSQ` optimizer |
| `scipy.signal` | https://docs.scipy.org/doc/scipy/reference/signal.html | `cont2discrete`, `place_poles`, `solve_ivp` |
| `scipy.linalg` | https://docs.scipy.org/doc/scipy/reference/linalg.html | `solve_continuous_are`, `solve_discrete_are` |
| `marimo` | https://docs.marimo.io | `@app.cell`, `mo.md()`, `mo.ui.*` |
| `numpy` | https://numpy.org/doc/stable/ | Prefer `np.trapezoid` over deprecated `np.trapz` |
| `polars` | https://docs.pola.rs | Used for data storage when needed (not pandas) |
| `rich` | https://rich.readthedocs.io | Used in `utils/logging.py` |
| `uv` | https://docs.astral.sh/uv/ | Package manager; prefer over pip |

---

## Coding conventions

- **Type annotations required** on all public functions and class methods.
- **Dataclasses** for plain data containers (`LTIModel`, `SimulationResult`).
- **`__post_init__`** for validation / normalisation in dataclasses.
- **`from __future__ import annotations`** at the top of every module.
- **NumPy arrays** as the universal numeric type; avoid raw lists for matrices.
- **`np.atleast_1d` / `np.atleast_2d`** before shape-dependent operations.
- **No lógica de pesquisa nos notebooks** — notebooks only *orchestrate* calls to `src/`.
- **Seeds always set** before any random operation in experiments and tests.
- `experiments/results/` is **never committed** — reproduce from config + seed.

---

## Adding a new controller

1. Implement `compute(x, r, t) -> np.ndarray` and `reset() -> None`.
2. Verify it satisfies `ControllerProtocol` (duck-typed; no explicit `isinstance` needed).
3. Add a test in `tests/test_<name>.py`.
4. Optionally register it in `experiments/run_experiment.py` for CLI use.

## Adding a new simulation backend

1. Implement `simulate(model, controller, x0, r_func, t_span, dt) -> SimulationResult`.
2. Handle both `LTIModel` and `NonlinearModel` (or raise `NotImplementedError` with a clear message).
3. Use a `try/except ImportError` guard if the backend requires optional dependencies.
4. Add integration tests in `tests/test_backends.py`.

## Adding a new experiment config

Create `experiments/configs/<name>.yaml` following the schema in `demo_a.yaml`.
Run with `uv run python -m control_lab.experiments.run_experiment --config experiments/configs/<name>.yaml`.
