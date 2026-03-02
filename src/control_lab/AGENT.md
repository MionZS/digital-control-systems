# AGENT.md — `src/control_lab/`

> For global context see `AGENT.md` at the repo root and `.github/copilot-instructions.md`.

## Package overview

`control_lab` is installed as a `src`-layout package (`uv sync`).
Top-level `__init__.py` re-exports the most-used public symbols:

```python
from control_lab import LTIModel, NonlinearModel, PIDController, LQRController,
                        lqr_continuous, lqr_discrete, SimulationResult,
                        compute_metrics, ControlBackend
```

The `interfaces.py` module defines `Protocol` classes that are the **contracts** between layers.
Never import concrete implementations across layers — depend on the protocol.

## Layer dependency rules

```
experiments  →  sim  →  models
                ↑           ↑
              design       ident
                ↑
              utils
```

- `models/` has no dependencies on other `control_lab` sub-packages.
- `design/` may import from `models/` but not from `sim/` or `ident/`.
- `sim/` imports from `models/` and may depend on `design/` via the `ControllerProtocol`.
- `ident/` imports from `models/` only.
- `utils/` has no internal dependencies.
- `experiments/` may import from any layer.

## Sub-package AGENT.md files

Each sub-package has its own `AGENT.md` with focused instructions:

| Directory | What it contains |
|---|---|
| `models/` | LTI and nonlinear system models, dataset generators |
| `design/` | PID, LQR, observers, discretisation |
| `sim/` | Simulation backends, result dataclass, metrics |
| `ident/` | SINDy wrapper, feature libraries, validation |
| `utils/` | Logging, plotting, seeding |
| `experiments/` | CLI runner, YAML config schema |

## Conventions in this package

- Every module starts with `from __future__ import annotations`.
- Public classes and functions have type annotations on all parameters and return types.
- Use `np.atleast_1d` / `np.atleast_2d` to normalise inputs before shape-dependent code.
- Dataclasses use `__post_init__` to coerce inputs to `np.ndarray` with `dtype=float`.
- Raise `ValueError` (not `AssertionError`) for invalid arguments.
- Raise `RuntimeError` for operations that require prior setup (e.g., calling `predict` before `fit`).
