# Step-Response Input

Put your impulse-response input files in this folder.

Expected format:

- CSV format (header required): time column `t` and output column `y`
- TXT format (whitespace separated):
  `time_s internal_temp control_signal ambient_temp`

The notebook estimates a second-order model and reports:

- gain `K`
- damping ratio `ζ`
- natural frequency `ω_n`
- delay `θ`
- overshoot, rise time, settling time

Example:

```csv
t,y
0.00,0.00
0.01,0.75
0.02,0.63
```

Notebook defaults are stored at `input/impulse_response/notebook_defaults.json`.

Sample TXT file for thermal step data:

- `sample_thermal_step.txt`

Run the detector entrypoint:

```bash
uv run python Scripts/run_impulse.py
```

Bypass the TUI and run immediately:

```bash
uv run python Scripts/run_impulse.py run -i input/impulse_response/sample_thermal_step.txt
```

For two-file ZOH identification (Notebook 05), use:

- `sample_control.csv` (control signal)
- `sample_output.csv` (output signal)

Notebook 05 defaults file:

- `input/impulse_response/zoh_defaults.json`
