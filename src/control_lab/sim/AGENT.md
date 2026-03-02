# AGENT.md — `src/control_lab/sim/`

> For package-level context see `src/control_lab/AGENT.md`.

## What lives here

Simulation backends and result types.

| File | Purpose |
|---|---|
| `common.py` | `SimulationResult` dataclass + `compute_metrics()` |
| `backend_control.py` | `ControlBackend` — scipy RK45 for LTI/nonlinear; algebraic for discrete LTI |
| `backend_collimator.py` | Optional `CollimatorBackend` (guarded `try/except ImportError`) |

## `SimulationResult` — key facts

```python
@dataclass
class SimulationResult:
    t: np.ndarray          # shape (N,)
    x: np.ndarray          # shape (N, n_states)
    y: np.ndarray          # shape (N, n_outputs)
    u: np.ndarray          # shape (N, n_inputs)
    metadata: dict         # free-form run metadata
```

All array shapes are `(N, *)` — time is the first axis.

## `compute_metrics()` — key facts

Returns a dict with these keys:

| Key | Description |
|---|---|
| `overshoot` | Percentage overshoot relative to `r_final` |
| `settling_time` | First time the output enters and stays in the 2 % band |
| `iae` | Integral absolute error `∫|e|dt` |
| `ise` | Integral squared error `∫e²dt` |
| `control_effort` | `∫u²dt` |

Uses `np.trapezoid` (NumPy ≥ 2.0) with a fallback shim for older versions.

## `ControlBackend` — key facts

The `simulate(model, controller, x0, r_func, t_span, dt)` method:

1. Builds a time grid `np.arange(t0, tf + dt/2, dt)`.
2. At each step:
   - Reads the current state `x`.
   - Calls `controller.compute(x, r_func(t), t)` for the control input `u`.
   - Computes output `y = Cx + Du` (LTI) or `g(x, u, t)` (nonlinear).
   - Advances the state:
     - **Discrete LTI** (`model.dt is not None`): `x = Ad x + Bd u` (algebraic).
     - **Continuous LTI**: `solve_ivp(A x + B u, [t, t+dt], x, method='RK45')`.
     - **Nonlinear**: `solve_ivp(f(x, u, t), [t, t+dt], x, method='RK45')`.
3. Returns a `SimulationResult`.

The controller is called with the **full state** `x`, not just the output `y`.
If the controller needs only the measured output, it must slice `x[0]` internally.

## `backend_collimator.py` — key facts

- Wraps `pycollimator` which is an optional extra (`uv add pycollimator --optional collimator`).
- File structure: `try: import collimator ... except ImportError: CollimatorBackend = None`.
- Export `None` when unavailable; callers check `if CollimatorBackend is not None`.

## When adding a new backend

1. Implement `simulate(model, controller, x0, r_func, t_span, dt) -> SimulationResult`.
2. Use `isinstance(model, LTIModel)` / `isinstance(model, NonlinearModel)` for dispatch.
3. Wrap optional dependency imports in `try/except ImportError`.
4. Add integration tests in `tests/test_backends.py` with `pytest.importorskip(...)`.

## References

- `scipy.integrate.solve_ivp`: https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html
- `pycollimator` docs: https://docs.collimator.ai/
