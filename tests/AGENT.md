# AGENT.md — `tests/`

> For global context see `AGENT.md` at the repo root.

## What lives here

The pytest test suite. Run with:
```bash
uv run pytest tests/ -v --tb=short
```

| File | Tests |
|---|---|
| `test_lti.py` | `LTIModel` creation, discretisation, stability, poles |
| `test_pid.py` | `PIDController` step response, saturation, anti-windup, reset |
| `test_lqr.py` | `lqr_continuous/discrete`, `LQRController.compute` |
| `test_sindy.py` | `SINDyIdentifier` fit, predict shape, rollout error |
| `test_backends.py` | `ControlBackend` with LTI and nonlinear models, `compute_metrics` |

## Conventions

### Optional dependencies — always use `importorskip`
Tests that require optional packages must skip gracefully:
```python
ps = pytest.importorskip("pysindy")
control = pytest.importorskip("control")
```
Place the `importorskip` at the **top of the test function** (not module level) so other
tests in the file still run when the dep is absent.

### Seeds — always set them
Any test that uses random data or random operations must call `set_global_seed(seed)`
at the start:
```python
from control_lab.utils.seeding import set_global_seed
set_global_seed(42)
```

### Assertions
- Use `pytest.approx` for floating-point comparisons.
- Prefer `assert condition, f"message: {actual_value}"` over bare `assert condition`.
- Test invariants, not implementation details:
  - ✅ `assert model.is_stable()`
  - ❌ `assert model._internal_list[0] == ...`

### Test naming
- Files: `test_<module>.py`
- Functions: `test_<what_it_tests>()`
- Add a brief docstring explaining the scenario being tested.

## When adding new source code

Add a corresponding test. Minimum coverage per new class:
- Constructor (happy path + error path).
- Main public method(s).
- Any edge case called out in the source docstring.

## When modifying existing source code

Run only the relevant test file first (faster feedback), then the full suite:
```bash
uv run pytest tests/test_lti.py -v          # targeted
uv run pytest tests/ -v --tb=short          # full suite
```

## CI matrix

Tests run on Python 3.12 and 3.13 in CI (`.github/workflows/ci.yml`).
Python 3.14 gets a smoke import only (`continue-on-error: true`).
