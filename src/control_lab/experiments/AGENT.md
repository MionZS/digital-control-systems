# AGENT.md — `src/control_lab/experiments/`

> For package-level context see `src/control_lab/AGENT.md`.

## What lives here

CLI entry point and YAML experiment configurations.
This is the **top layer** — it may import from any other sub-package.

| File | Purpose |
|---|---|
| `run_experiment.py` | CLI runner: loads config, runs experiment, saves artefacts |
| `configs/demo_a.yaml` | Config for Demo A (mass-spring-damper + PID) |

## `run_experiment.py` — key facts

### CLI usage
```bash
uv run python -m control_lab.experiments.run_experiment --config experiments/configs/demo_a.yaml
```

### What it does
1. Loads the YAML config.
2. Sets the global seed (`set_global_seed(seed)`).
3. Instantiates the model (`model.type`).
4. Instantiates the controller (`controller.type`).
5. Runs the simulation via `ControlBackend`.
6. Computes metrics.
7. Saves artefacts to `experiments/results/YYYYMMDD-HHMM__<name>__seed<N>/`.

### Output artefacts
| File | Description |
|---|---|
| `metrics.json` | Step-response performance metrics |
| `config_used.yaml` | Frozen copy of the config that produced this run |
| `trajectory.csv` | Full state/output/input time series |

`experiments/results/` is **gitignored**. Reproduce any run with the same config + seed.

## YAML config schema

```yaml
name: <str>          # used in the output directory name
seed: <int>          # random seed (reproducibility)

model:
  type: mass_spring_damper   # supported: mass_spring_damper
  m: <float>                 # mass (kg)
  k: <float>                 # spring constant (N/m)
  c: <float>                 # damping (N·s/m)
  dt: null                   # null = continuous; float = discrete

controller:
  type: pid | lqr
  # PID fields:
  kp, ki, kd: <float>
  dt: <float>
  u_min, u_max: <float>      # optional saturation limits
  # LQR fields:
  Q: [[...]]                 # state cost matrix (list of lists)
  R: [[...]]                 # input cost matrix

simulation:
  t_span: [t0, tf]
  dt: <float>
  x0: [x1, x2, ...]
  r: <float>                 # constant step reference
```

## When adding a new model or controller type

1. Add a new `elif model_cfg["type"] == "..."` block in `run_experiment.py`.
2. Import the new class locally inside the block (not at the top of the file).
3. Add a corresponding YAML config in `experiments/configs/`.
4. Keep the YAML schema self-documenting with inline comments.

## Reproducing a published result

```bash
uv run python -m control_lab.experiments.run_experiment \
    --config path/to/config_used.yaml  # use the frozen config from the results dir
```

Because `config_used.yaml` is saved with every run, any result can be exactly reproduced
as long as the codebase state is the same (use git tags / commits for this).
