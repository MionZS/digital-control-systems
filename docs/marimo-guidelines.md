# Marimo Notebook Guidelines

This repository uses Marimo `.py` notebooks (not `.ipynb`).

## Core syntax rules

1. Each cell must be a function decorated with `@app.cell`.
2. The function body is treated as notebook cell code; avoid control-flow `return` statements inside branches.
3. Use a single `return` at the end of the cell to export names to downstream cells.
4. If a cell is conditional, assign fallback values and still return at the end.

Bad pattern (can trigger "return outside function" during Marimo compilation):

```python
@app.cell
def _(x):
    if x is None:
        return
    y = x + 1
    return y
```

Good pattern:

```python
@app.cell
def _(x):
    y = None
    if x is not None:
        y = x + 1
    return (y,)
```

## Dependency flow

- Downstream cells must receive needed names as function arguments.
- Upstream cells must return every name they export.
- Do not rely on hidden state.

## Project conventions

- Keep algorithm logic in `src/control_lab/*`; notebooks orchestrate calls.
- Prefer `matplotlib.use("Agg")` in import cells.
- Close figures with `plt.close("all")` after creating plotted outputs.
- Keep file defaults in `input/*/*.json` and load them in notebook setup cells.

## Useful commands

```bash
uv run marimo edit notebooks/05_zoh_two_file_identification.py
uv run marimo run notebooks/05_zoh_two_file_identification.py
uv run pytest
uv run ruff check .
```

## References

- https://docs.marimo.io/
- https://docs.marimo.io/guides/reactivity/
- https://docs.marimo.io/guides/lint_rules/rules/invalid_syntax/
