# AGENT.md — `src/control_lab/models/`

> For package-level context see `src/control_lab/AGENT.md`.

## What lives here

System model definitions — the **lowest layer** of the stack.
Nothing in this package imports from other `control_lab` sub-packages.

| File | Purpose |
|---|---|
| `lti.py` | `LTIModel` dataclass — continuous and discrete state-space |
| `nonlinear.py` | `NonlinearModel` dataclass — callable dynamics `f(x,u,t)` / `g(x,u,t)` |
| `datasets.py` | Signal generators for identification: PRBS, multi-sine |

## `LTIModel` — key facts

- Continuous: `ẋ = Ax + Bu`, `y = Cx + Du` (`dt = None`)
- Discrete: `x[k+1] = Ax[k] + Bu[k]`, `y[k] = Cx[k] + Du[k]` (`dt = float`)
- `__post_init__` coerces all matrices to `np.ndarray` with `dtype=float` via `np.atleast_2d`.
- `to_discrete_time(dt, method)` uses `scipy.signal.cont2discrete` (ZOH by default).
- `to_continuous_time()` recovers `Ac` via matrix logarithm — use with caution on noisy data.
- `is_stable()` checks `Re(poles) < 0` for continuous, `|poles| < 1` for discrete.

### Factory methods
- `LTIModel.mass_spring_damper(m, k, c, dt=None)` — canonical 2-state LTI example.

### When adding a new factory method
Add it as a `@classmethod` that returns `"LTIModel"` and include:
1. Physical parameters as arguments with type annotations.
2. A docstring describing the state and input definitions.
3. A test in `tests/test_lti.py`.

## `NonlinearModel` — key facts

- Stores `f_func: Callable[[ndarray, ndarray, float], ndarray]` (dynamics).
- Stores `g_func: Callable[[ndarray, ndarray, float], ndarray]` (output map).
- `n_states`, `n_inputs`, `n_outputs` are explicit integer fields (not inferred).
- Factory: `NonlinearModel.inverted_pendulum(m, l, g=9.81, b=0.1)`.

### When adding a new nonlinear model
1. Define `f` and `g` as closures inside the classmethod (capture params).
2. Always include `n_states`, `n_inputs`, `n_outputs` in the constructor call.
3. Document state vector convention (e.g., `[theta, theta_dot]`) in the docstring.

## `datasets.py` — key facts

- `generate_prbs(n_steps, dt, amplitude, seed)` → `(t, u)` as 1-D arrays.
- `generate_multisine(freqs, amplitudes, t_grid, seed)` → `u` 1-D array.
- These signals are used as open-loop excitation for SINDy identification.

## Do NOT add here

- Controller logic (→ `design/`)
- Simulation logic (→ `sim/`)
- Identification logic (→ `ident/`)

## References

- State-space representation: https://python-control.readthedocs.io/en/latest/intro.html
- `scipy.signal.cont2discrete`: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.cont2discrete.html
