# AGENT.md — Repository root

> **Start here.** This file gives AI coding agents a map of the repository and top-level
> conventions. For global Copilot context see `.github/copilot-instructions.md`.

## Repository overview

`control-lab` is a modular Python control-systems framework.
Primary Python version: **3.12**. Package manager: **uv**.

```
.github/
  copilot-instructions.md   ← global Copilot context
  workflows/ci.yml           ← lint + test matrix (3.12, 3.13) + smoke-3.14

src/control_lab/            ← all importable source code (see AGENT.md inside)
tests/                      ← pytest suite (see AGENT.md inside)
notebooks/                  ← Marimo .py notebooks (see AGENT.md inside)
experiments/
  configs/                  ← YAML experiment configs (source of truth for runs)
  results/                  ← gitignored; reproduce via config + seed

pyproject.toml              ← dependencies, ruff, mypy, pytest config
uv.lock                     ← lockfile (commit this)
```

## Key entry points

| Task | Command |
|---|---|
| Install | `uv sync --group dev` |
| Run tests | `uv run pytest tests/ -v` |
| Lint | `uv run ruff check .` |
| Format | `uv run ruff format .` |
| Type-check | `uv run mypy src/` |
| Run Demo A | `uv run python -m control_lab.experiments.run_experiment --config experiments/configs/demo_a.yaml` |
| Open notebook | `uv run marimo edit notebooks/01_python_control_basics.py` |

## Invariants to maintain

- `experiments/results/` is **never committed** (`.gitignore`).
- `uv.lock` **is committed** (reproducibility).
- All public APIs carry type annotations.
- Optional dependencies (`pysindy`, `pycollimator`) are imported **lazily** inside
  the function that needs them, never at module top level.
- Notebooks are `.py` (Marimo), never `.ipynb`.
- Uppercase matrix names (A, B, C, D, K, Q, R) are allowed — ruff ignores are set.

## When making changes

1. Check the module-level `AGENT.md` in the relevant directory first.
2. Run `uv run pytest tests/ -v --tb=short` after any code change.
3. Run `uv run ruff check .` before committing.
4. If you add a dependency, run `uv add <pkg>` (not `pip install`).
5. If you change a public interface, update the corresponding `Protocol` in
   `src/control_lab/interfaces.py` and check all implementations.
