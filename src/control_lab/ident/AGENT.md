# AGENT.md — `src/control_lab/ident/`

> For package-level context see `src/control_lab/AGENT.md`.

## What lives here

Data-driven system identification using **SINDy** (Sparse Identification of Nonlinear Dynamics).
`pysindy` is imported lazily inside function bodies — this module is importable even when
`pysindy` is not installed.

| File | Purpose |
|---|---|
| `feature_library.py` | Thin wrappers: `PolynomialLibrary(degree)`, `FourierLibrary(n_freqs)` |
| `sindy_fit.py` | `SINDyIdentifier` — fit, predict, get_equations |
| `sindy_validate.py` | `one_step_error`, `rollout_error`, `plot_validation` |

## `SINDyIdentifier` — key facts

```python
identifier = SINDyIdentifier(feature_library=None, optimizer=None, dt=0.01)
identifier.fit({"x": x_arr, "u": u_arr, "t": t_arr})  # t is optional; uses self.dt
x_pred = identifier.predict(x0, u_seq, t_grid)          # returns shape (N, n_states)
eqs = identifier.get_equations()                         # list of human-readable strings
model = identifier.model                                  # underlying ps.SINDy object
```

### `fit(data)` — data dict schema

| Key | Required | Shape | Description |
|---|---|---|---|
| `'x'` | ✅ | `(N, n_states)` | State trajectory |
| `'u'` | ❌ | `(N, n_inputs)` | Control input (omit for autonomous systems) |
| `'t'` | ❌ | `(N,)` | Time array; falls back to `self.dt` if absent |

### Lazy import pattern
```python
def fit(self, data):
    import pysindy as ps  # imported here, NOT at module top
    ...
```
Always follow this pattern when adding new methods that need pysindy.

## `feature_library.py` — key facts

- `PolynomialLibrary(degree)` wraps `ps.PolynomialLibrary(degree=degree)`.
- `FourierLibrary(n_freqs)` wraps `ps.FourierLibrary(n_frequencies=n_freqs)`.
- Both functions do a lazy `import pysindy as ps` inside the function body.
- To add a new library, follow the same pattern and add a test.

## Validation utilities

- `one_step_error(identifier, data)` — computes mean |ẋ_pred - ẋ_true| using
  `model.predict()` and `model.differentiate()` from pysindy.
- `rollout_error(identifier, data)` — simulates from `x0` over the full `t_grid` and
  computes RMSE against ground truth.
- `plot_validation(identifier, data, title)` — matplotlib figure comparing true vs predicted
  trajectories per state dimension. Returns the `Figure`.

## Hybrid physics + data workflow

The recommended workflow for a known physical system:
1. Simulate with the nominal model to collect data.
2. Add measurement noise to `x` and `u`.
3. Fit SINDy to the **residuals** (not raw states) if the base model is known.
4. Validate on a held-out trajectory (different initial condition or reference).

## Do NOT add here

- Controller design (→ `design/`)
- Model definitions (→ `models/`)
- Simulation loops (→ `sim/`)

## References

- pysindy documentation: https://pysindy.readthedocs.io/
- SINDy paper: https://doi.org/10.1073/pnas.1517384113
- pysindy feature libraries: https://pysindy.readthedocs.io/en/latest/api/feature_library/
- pysindy optimisers: https://pysindy.readthedocs.io/en/latest/api/optimizers/
