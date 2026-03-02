# control-lab

**A modular Python framework for digital control-systems simulation, design, and identification.**

[![CI](https://github.com/digital-control-systems/digital-control-systems/actions/workflows/ci.yml/badge.svg)](https://github.com/digital-control-systems/digital-control-systems/actions/workflows/ci.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/)

---

## Architecture

```
control-lab
├── models/       — LTI (state-space) and nonlinear system models
├── design/       — PID, LQR, observer design, discretisation helpers
├── sim/          — Simulation backends (scipy / collimator)
├── ident/        — SINDy-based system identification
├── utils/        — Logging, plotting, random-seed utilities
└── experiments/  — CLI runner + YAML experiment configs
```

Dependency layers:

```
experiments  ──→  sim  ──→  models
                  ↑           ↑
               design       ident
                  ↑
               utils
```

---

## Quick start

### Install with `uv`

```bash
git clone https://github.com/digital-control-systems/digital-control-systems.git
cd digital-control-systems

# Install with dev dependencies
uv sync --group dev

# Optional: include the collimator backend
uv sync --extra collimator --group dev
```

### Run Demo A (mass-spring-damper + PID)

```bash
uv run python -m control_lab.experiments.run_experiment \
    --config experiments/configs/demo_a.yaml
```

Results land in `experiments/results/YYYYMMDD-HHMM__demo_a__seed42/`.

### Run the test suite

```bash
uv run pytest tests/ -v --tb=short
```

### Lint

```bash
uv run ruff check .
```

---

## Usage examples

```python
from control_lab.models.lti import LTIModel
from control_lab.design.lqr import LQRController, lqr_continuous
from control_lab.sim.backend_control import ControlBackend
from control_lab.sim.common import compute_metrics
import numpy as np

model = LTIModel.mass_spring_damper(m=1.0, k=1.0, c=0.5)
print("Poles:", model.poles())          # [-0.25±0.97j]
print("Stable:", model.is_stable())     # True

Q, R = np.diag([100.0, 1.0]), np.array([[0.1]])
K, _, _ = lqr_continuous(model.A, model.B, Q, R)
ctrl = LQRController(K)

backend = ControlBackend()
result = backend.simulate(
    model, ctrl,
    x0=np.zeros(2),
    r_func=lambda t: np.array([1.0, 0.0]),
    t_span=(0.0, 10.0),
    dt=0.01,
)
print(compute_metrics(result, r_final=1.0))
```

---

## Project structure

```
digital-control-systems/
├── pyproject.toml
├── README.md
├── src/control_lab/
│   ├── __init__.py
│   ├── interfaces.py
│   ├── models/         lti.py · nonlinear.py · datasets.py
│   ├── design/         pid.py · lqr.py · observers.py · discretization.py
│   ├── sim/            common.py · backend_control.py · backend_collimator.py
│   ├── ident/          sindy_fit.py · sindy_validate.py · feature_library.py
│   ├── utils/          logging.py · plotting.py · seeding.py
│   └── experiments/    run_experiment.py · configs/demo_a.yaml
├── tests/
├── notebooks/          01_python_control_basics · 02_sindy_from_sim · 03_compare
└── .github/workflows/  ci.yml
```

---

## Roadmap

| Sprint | Theme | Status |
|--------|-------|--------|
| 1 | Foundation — LTI, PID, LQR, scipy backend | ✅ |
| 2 | Identification — SINDy, datasets, observers | ✅ |
| 3 | Advanced — MPC stub, nonlinear observer, discrete LQR | 🔜 |
| 4 | Deployment — Collimator backend, Polars store, dashboard | 🔜 |

---

## Contributing

1. Fork and create a feature branch.
2. `uv sync --group dev` to install dev dependencies.
3. Add tests under `tests/`.
4. Ensure `uv run ruff check .` is clean.
5. Open a pull request.

---

## License

MIT
