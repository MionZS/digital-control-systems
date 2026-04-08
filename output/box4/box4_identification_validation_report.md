# Box4 Identification and Validation Report (Didactic)

## 1) Objective and workflow

This report documents the complete process used to:

1. Identify a dynamic model from `input/EstimacaoBox4` (estimation experiment).
2. Reuse the identified model on `input/ValidacaoBox4` (validation experiment).
3. Compare simulated model output against measured lab output.

Main principle:

- Estimation data is used to fit model parameters.
- Validation data is never used for fitting, only for testing generalization.

This is exactly the identification/validation split typically taught in control-system modeling classes.

---

## 2) "No axis warping" policy

From now on, all comparison plots are generated with a single y-axis per subplot for compared signals.

- No dual-axis scaling (`twinx`) is used in overlays.
- No manual amplitude compression (`set_ylim`) is applied to force visual alignment.
- The viewer sees raw amplitude mismatch directly.

Why this matters:

- Dual-axis plots can visually hide gain errors.
- Shared-axis plots make steady-state and dynamic mismatch obvious.

---

## 3) Tools and libraries used

## 3.1 CLI and environment

- `uv` for execution and dependency resolution.
- `python` scripts in `Scripts/` for reproducible batch runs.
- `ruff` for lint/static style checks.

## 3.2 Numerical and plotting libraries

- `numpy`: vectors, statistics, residual computations.
- `scipy.signal`: transfer-function discretization and discrete simulation (`cont2discrete`, `lfilter`).
- `scipy.optimize`: nonlinear least-squares fitting in identification.
- `matplotlib`: figure generation for diagnostics and validation.

## 3.3 Relation to the control_lab package

The repository `src/control_lab/` provides the modular architecture for control work (models/design/sim/ident).
The two scripts used here are standalone experiment utilities that apply the same control concepts:

- signal parsing and preprocessing,
- model structure selection,
- continuous and discrete transfer-function handling,
- simulation and objective metrics.

In other words, scripts are experiment runners; `control_lab` is the reusable framework layer.

---

## 4) Scripts and function groups

## 4.1 Identification script

File: `Scripts/automatic_order_identification.py`

Function groups:

1. Data loading and alignment
   - `read_lvm`: parses LabVIEW `.lvm` numeric columns.
   - `align_timebase`: enforces common time base and computes `dt`.

2. Step-region extraction
   - `detect_steps`: finds control transitions.
   - `build_regions`: builds local regions with initial/final plateaus.
   - `normalized_regions`: rescales by each input step for fitting.

3. Candidate-model fitting and selection
   - `fit_candidate`: fits first/second/third-order candidates.
   - `choose_model`: applies score-based model selection.

4. Model-domain conversion and simulation helpers
   - `discrete_from_continuous`: computes `G(z)` from `G(s)` using ZOH.
   - `simulate_output_from_input`: simulates output from full input trace.

5. Output writers
   - `save_plot`: diagnostic figure.
   - `save_overlay_plots`: overlay figures with shared axis (no warping).

## 4.2 Validation script

File: `Scripts/validate_identified_model.py`

Function groups:

1. Data and model loading
   - `read_lvm`, `align_timebase`.
   - JSON model load from estimation output.

2. Discrete simulation with delay handling
   - `infer_delay_samples`: converts identified delay to sample delay.
   - `simulate_discrete`: runs `lfilter` and applies delay shift.

3. Error metrics
   - `calc_metrics`: RMSE, MAE, NRMSE, fit %, max-abs-error.

4. Validation visualization
   - `save_validation_plot`: input, measured-vs-model, and error subplots.

---

## 5) Step-by-step execution (what was run)

## Step A: Estimate model from Box4 estimation data

Command:

```bash
uv run python Scripts/automatic_order_identification.py \
  --control input/EstimacaoBox4/controle.lvm \
  --output input/EstimacaoBox4/saida.lvm \
  --out-json output/box4/identified_model.json \
  --out-plot output/box4/identified_model_plot.png \
  --time-scale 0.1
```

Observed selection result:

- apparent family: `third_or_higher_like`
- selected model: `second_order_underdamped`
- order: `2`
- RMSE: `0.238337`
- BIC: `-2136.077`

Generated artifacts:

- `output/box4/identified_model.json`
- `output/box4/identified_model_plot.png`
- `output/box4/identified_model_plot_control_overlays.png`

## Step B: Validate identified model on Box4 validation data

Command:

```bash
uv run python Scripts/validate_identified_model.py \
  --model-json output/box4/identified_model.json \
  --control input/ValidacaoBox4/controle.lvm \
  --output input/ValidacaoBox4/saida.lvm \
  --out-json output/box4/validation_results.json \
  --out-plot output/box4/validation_plot.png \
  --time-scale 0.1
```

Observed validation result:

- RMSE: `1.657587`
- MAE: `1.302044`
- NRMSE: `0.220959`
- Fit %: `-11.361`
- Max abs error: `4.712380`

Generated artifacts:

- `output/box4/validation_results.json`
- `output/box4/validation_plot.png`

---

## 6) Intermediate values used for interpretation

These were extracted from the measured data as segment-end values (last samples in each plateau), so each segment reflects "initial-to-final" behavior before next step.

## 6.1 Estimation experiment (`input/EstimacaoBox4`)

| Segment | Time window [s] | Control level | Measured final output |
|---:|---:|---:|---:|
| 1 | 0.0 to 6.7 | 0.0 | 0.005227 |
| 2 | 6.8 to 11.7 | 4.0 | 2.342243 |
| 3 | 11.8 to 14.5 | 6.0 | 3.774151 |
| 4 | 14.6 to 20.4 | 2.0 | 2.871210 |
| 5 | 20.5 to 24.3 | -1.0 | 1.016893 |
| 6 | 24.4 to 28.6 | -4.0 | -1.759739 |
| 7 | 28.7 to 38.7 | 5.0 | 4.104532 |
| 8 | 38.8 to 44.2 | 3.0 | 3.570658 |
| 9 | 44.3 to 47.1 | 4.0 | 3.674030 |
| 10 | 47.2 to 52.1 | -2.0 | 0.465492 |
| 11 | 52.2 to 55.7 | -1.0 | -0.478810 |
| 12 | 55.8 to 60.1 | -2.0 | -1.342811 |
| 13 | 60.2 to 63.6 | -4.0 | -2.548793 |
| 14 | 63.7 to 68.4 | -6.0 | -4.697547 |
| 15 | 68.5 to 74.6 | 5.0 | 1.767708 |
| 16 | 74.7 to 98.8 | 0.0 | 0.020586 |

## 6.2 Validation experiment (`input/ValidacaoBox4`)

| Segment | Time window [s] | Control level | Measured final output |
|---:|---:|---:|---:|
| 1 | 0.0 to 3.9 | 0.0 | -0.005798 |
| 2 | 4.0 to 10.7 | 3.0 | 2.194514 |
| 3 | 10.8 to 14.1 | 5.0 | 3.464163 |
| 4 | 14.2 to 16.9 | -3.0 | 1.734504 |
| 5 | 17.0 to 21.1 | -6.0 | -2.646428 |
| 6 | 21.2 to 26.2 | 7.0 | 2.704809 |
| 7 | 26.3 to 30.5 | 3.0 | 3.419042 |
| 8 | 30.6 to 33.8 | -4.0 | 0.549235 |
| 9 | 33.9 to 37.5 | 6.0 | 2.358813 |
| 10 | 37.6 to 41.4 | 2.0 | 2.700029 |
| 11 | 41.5 to 44.8 | -5.0 | -0.416927 |
| 12 | 44.9 to 47.8 | 4.0 | 0.385446 |
| 13 | 47.9 to 51.6 | -1.0 | 0.231154 |
| 14 | 51.7 to 55.2 | 9.0 | 3.804487 |
| 15 | 55.3 to 58.5 | -2.0 | 2.385835 |
| 16 | 58.6 to 61.8 | 3.0 | 2.031108 |
| 17 | 61.9 to 66.6 | -5.0 | -1.793835 |
| 18 | 66.7 to 69.0 | 7.0 | -0.163023 |
| 19 | 69.1 to 71.3 | 4.0 | 2.003577 |
| 20 | 71.4 to 74.0 | 2.0 | 2.364549 |
| 21 | 74.1 to 77.2 | -3.0 | 0.368175 |
| 22 | 77.3 to 81.3 | 4.0 | 1.739858 |
| 23 | 81.4 to 98.3 | 0.0 | 0.069850 |

Interpretation:

- Validation input changes sign and magnitude frequently.
- This excites dynamics beyond a simple second-order underdamped approximation fitted on the estimation data.
- Result: low transferability on this validation sequence (negative fit %).

---

## 7) Identified model details used in validation

From `output/box4/identified_model.json`:

- Model: second-order underdamped
- Continuous form:
  - Numerator: `[0.3280847330378226]`
  - Denominator: `[1.0, 0.5677576168815844, 1.0657185605205377]`
- Discrete form (ZOH, dt = 0.1 s):
  - Numerator: `[0.0, 0.0016083901989587535, 0.0015782273316758966]`
  - Denominator: `[1.0, -1.934454808614382, 0.944805907296215]`
- Applied delay in validation: `5` samples

---

## 8) Theory section (aligned with standard control slides)

## 8.1 Identification from step-like experiments

Typical slides present this sequence:

1. Excite plant with known input changes.
2. Observe output transients and steady states.
3. Estimate gain, dominant time constants, damping, and delay.
4. Select simplest model that explains data.

This is exactly what Step A does with candidate models of increasing complexity.

## 8.2 First-order vs second-order interpretation

Slides usually connect response shape to model class:

- First-order: monotonic rise/decay, one dominant pole.
- Second-order underdamped: oscillatory tendencies, where damping ratio governs overshoot/settling tradeoff.
- Higher-order: richer dynamics or unmodeled nonlinearities.

The estimator selected second-order underdamped for Box4 estimation data.

## 8.3 Discretization for digital simulation

A continuous model `G(s)` must be discretized to run with sampled data:

- method: Zero-Order Hold (ZOH), matching DAC/hold behavior in digital control loops.
- output: discrete `G(z)` filter coefficients.
- simulation: apply difference equation (`lfilter`) to control sequence.

This corresponds to the standard lecture transition from continuous plant model to sampled implementation.

## 8.4 Validation philosophy (generalization)

Slides on system identification emphasize:

- A model is useful only if it predicts unseen data.
- Good fit on estimation data is necessary but not sufficient.
- Validation mismatch indicates either model mismatch, changed operating conditions, nonlinear behavior, or insufficient model order/structure.

For Box4, validation fit is poor (negative fit %), so the current model is not robust for this validation trajectory.

---

## 9) Practical usage guide

## 9.1 Re-run identification

```bash
uv run python Scripts/automatic_order_identification.py \
  --control input/EstimacaoBox4/controle.lvm \
  --output input/EstimacaoBox4/saida.lvm \
  --out-json output/box4/identified_model.json \
  --out-plot output/box4/identified_model_plot.png \
  --time-scale 0.1
```

## 9.2 Re-run validation

```bash
uv run python Scripts/validate_identified_model.py \
  --model-json output/box4/identified_model.json \
  --control input/ValidacaoBox4/controle.lvm \
  --output input/ValidacaoBox4/saida.lvm \
  --out-json output/box4/validation_results.json \
  --out-plot output/box4/validation_plot.png \
  --time-scale 0.1
```

## 9.3 How to read the validation figure

- Top subplot: command profile used for test.
- Middle subplot: measured output (lab) vs model output (prediction).
- Bottom subplot: residual error over time.

Because we do not warp axes, any amplitude mismatch is directly visible.

---

## 10) Current conclusion for Box4

- Identification pipeline executed correctly.
- Validation pipeline executed correctly.
- Model transfer from EstimacaoBox4 to ValidacaoBox4 is weak (high RMSE, negative fit%).
- Next technical action should be model-structure improvement (e.g., higher-order, nonlinear, or operating-point dependent model), not plot rescaling.
