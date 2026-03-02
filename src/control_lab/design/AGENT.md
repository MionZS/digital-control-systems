# AGENT.md — `src/control_lab/design/`

> For package-level context see `src/control_lab/AGENT.md`.

## What lives here

Controller and observer design algorithms.
May import from `models/` but must **not** import from `sim/`, `ident/`, or `experiments/`.

| File | Purpose |
|---|---|
| `pid.py` | `PIDController` — discrete PID with anti-windup and derivative filter |
| `lqr.py` | `lqr_continuous`, `lqr_discrete` functions + `LQRController` class |
| `observers.py` | `luenberger_gain`, `kalman_gain`, `StateObserver` class |
| `discretization.py` | `discretize(A, B, dt, method)` + `compare_methods()` |

## `PIDController` — key facts

- **Discrete-time** only; sample time `dt` is required at construction.
- Anti-windup uses the **clamping** method: the integrator only accumulates when the output
  is not saturated (`u_min ≤ u_unsat ≤ u_max`).
- Derivative filter: first-order low-pass with time constant `tau` (default 0.1 s).
- `compute(x, r, t)` accepts scalars or arrays; uses the **first element** as the scalar
  measured output / reference.
- `reset()` zeroes `_integral`, `_prev_error`, `_d_filtered`.

### Tuning helpers
- `PIDController.tune_ziegler_nichols(ku, tu, dt)` classmethod for classic ZN tuning.

### When modifying PID
Always run `tests/test_pid.py` — the anti-windup and saturation tests are the key invariants.

## LQR functions — key facts

```python
K, P, E = lqr_continuous(A, B, Q, R)   # solves CARE: A^T P + PA - PBR^{-1}B^T P + Q = 0
K, P, E = lqr_discrete(A, B, Q, R)     # solves DARE
```

- `K` is the gain matrix: `u = -K x`.
- `LQRController` wraps `K` and implements `ControllerProtocol`: `u = -K (x - r)`.
- `LQRController.reset()` is a no-op (stateless).

### Q and R conventions
- `Q` penalises state deviation (shape `(n, n)`); typically diagonal.
- `R` penalises control effort (shape `(m, m)`); must be positive definite.
- Larger `Q[i,i]` → tighter tracking of state `i`; larger `R` → less aggressive control.

## `StateObserver` — key facts

- Euler-discretised `x̂_dot = Ax̂ + Bu + L(y - Cx̂)`.
- `update(y, u, dt)` takes measurements and returns the updated estimate.
- `reset()` zeroes the estimate.
- Use `luenberger_gain` or `kalman_gain` to compute `L` before constructing.

## `discretization.py` — key facts

- `discretize(A, B, dt, method='zoh')` wraps `scipy.signal.cont2discrete`.
- Supported methods: `'zoh'`, `'euler'`, `'tustin'`, `'bilinear'`, `'foh'`.
- `compare_methods(A, B, dt)` returns a dict keyed by method; safe (catches exceptions).

## Do NOT add here

- Simulation loops (→ `sim/`)
- Data-driven identification (→ `ident/`)
- Model definitions (→ `models/`)

## References

- LQR theory: https://python-control.readthedocs.io/en/latest/optimal.html
- `scipy.linalg.solve_continuous_are`: https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.solve_continuous_are.html
- `scipy.signal.place_poles`: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.place_poles.html
