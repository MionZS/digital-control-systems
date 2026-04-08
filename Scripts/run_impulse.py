from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(help="Entrypoint: run or open the impulse-response analysis notebook")

ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = ROOT / "input" / "impulse_response"
DEFAULTS_PATH = INPUT_DIR / "notebook_defaults.json"
DEFAULT_INPUT_CSV = INPUT_DIR / "sample_thermal_step.txt"
NOTEBOOK_PATH = ROOT / "notebooks" / "04_impulse_response_analysis.py"
NOTEBOOK_NAME = "04_impulse_response_analysis.py"
NOTEBOOKS = [(NOTEBOOK_NAME, NOTEBOOK_PATH)]


def _write_defaults(input_file: Path, dt: Optional[float], max_taps: Optional[int]) -> Path:
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    defaults = {
        "input_csv": str(input_file),
        "dt": dt if dt is not None else None,
        "max_taps": int(max_taps) if max_taps is not None else 300,
        "output_model_json": str(INPUT_DIR / "identified_model.json"),
    }
    DEFAULTS_PATH.write_text(json.dumps(defaults, indent=2), encoding="utf-8")
    return DEFAULTS_PATH


def _read_defaults() -> dict[str, object]:
    defaults = {
        "input_csv": str(DEFAULT_INPUT_CSV),
        "dt": None,
        "max_taps": 300,
        "output_model_json": str(INPUT_DIR / "identified_model.json"),
    }
    if DEFAULTS_PATH.exists():
        with DEFAULTS_PATH.open("r", encoding="utf-8") as fp:
            defaults.update(json.load(fp))
    return defaults


def _list_input_files() -> list[Path]:
    if not INPUT_DIR.exists():
        return []
    files = [
        path
        for path in sorted(INPUT_DIR.iterdir())
        if path.is_file() and path.suffix.lower() in {".csv", ".txt"}
    ]
    return files


def _choose_input_file() -> Path:
    files = _list_input_files()
    if not files:
        typer.echo(f"No input files found in {INPUT_DIR}")
        raise typer.Exit(1)

    default_file = DEFAULT_INPUT_CSV if DEFAULT_INPUT_CSV in files else files[0]
    default_index = files.index(default_file) + 1

    typer.echo("\nAvailable input files:")
    for idx, file_path in enumerate(files, 1):
        marker = " (default)" if file_path == default_file else ""
        typer.echo(f"  {idx}. {file_path.name}{marker}")

    selection = typer.prompt(
        "Select an input file by number", default=default_index, type=int, show_default=True
    )
    if not 1 <= selection <= len(files):
        raise typer.BadParameter(f"Selection must be between 1 and {len(files)}")

    selected_file = files[selection - 1]
    typer.echo(f"Selected input file: {selected_file.name}")
    return selected_file


def _show_current_config(defaults: dict[str, object], notebook_name: str) -> None:
    typer.echo("\nCurrent detector config:")
    typer.echo(f"  Notebook: {notebook_name}")
    typer.echo(f"  Input file: {defaults['input_csv']}")
    typer.echo(f"  Output model: {defaults['output_model_json']}")


def _choose_notebook() -> tuple[str, Path]:
    typer.echo("\nAvailable notebooks:")
    for idx, (name, _) in enumerate(NOTEBOOKS, 1):
        typer.echo(f"  {idx}. {name}")

    selection = typer.prompt("Select a notebook by number", default=1, type=int, show_default=True)
    if not 1 <= selection <= len(NOTEBOOKS):
        raise typer.BadParameter(f"Selection must be between 1 and {len(NOTEBOOKS)}")

    selected_name, selected_path = NOTEBOOKS[selection - 1]
    typer.echo(f"Selected notebook: {selected_name}")
    return selected_name, selected_path


def _run_notebook(mode: str, notebook_path: Path) -> None:
    if mode == "edit":
        typer.echo("Opening notebook in marimo editor...")
        subprocess.Popen(["uv", "run", "marimo", "edit", str(notebook_path)])
        return

    typer.echo("Running notebook headlessly (marimo run)...")
    rc = subprocess.run(["uv", "run", "marimo", "run", str(notebook_path)])
    if rc.returncode != 0:
        typer.echo("Notebook execution failed.")
        raise typer.Exit(rc.returncode)
    typer.echo(
        "Notebook finished. Identified model saved to input/impulse_response/identified_model.json"
    )


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    input_file: Optional[str] = typer.Option(
        None,
        "--input-file",
        "-i",
        help="Path to .csv (t,y) or .txt thermal step file",
    ),
    dt: Optional[float] = typer.Option(None, "--dt", help="Optional sampling time override"),
    max_taps: Optional[int] = typer.Option(None, "--max-taps", help="Maximum FIR taps"),
    mode: str = typer.Option("run", "--mode", "-m", help="Mode: 'run' (headless) or 'edit' (open)"),
    no_tui: bool = typer.Option(False, "--no-tui", help="Bypass prompts and use flags/defaults"),
):
    """Default entrypoint for the impulse-response detector notebook.

    Without subcommands, this opens a small TUI that shows the current config and
    then asks whether to edit or run the detector notebook.
    Use --no-tui to bypass prompts and go straight to the notebook.
    """
    if ctx.invoked_subcommand is not None:
        return

    selected_name, selected_path = NOTEBOOKS[0]
    if not no_tui:
        selected_name, selected_path = _choose_notebook()

    if input_file is None:
        input_file_path = _choose_input_file() if not no_tui else DEFAULT_INPUT_CSV
    else:
        input_file_path = Path(input_file)

    if not input_file_path.exists():
        typer.echo(f"Input file not found: {input_file_path}")
        raise typer.Exit(1)

    _write_defaults(input_file_path, dt, max_taps)
    typer.echo(f"Defaults written to: {DEFAULTS_PATH}")

    updated_defaults = _read_defaults()
    _show_current_config(updated_defaults, selected_name)

    if no_tui:
        _run_notebook(mode, selected_path)
    else:
        choice = typer.confirm("Open the notebook now?", default=True)
        if choice:
            selected_mode = mode
            if mode == "run":
                selected_mode = typer.prompt("Choose mode [run/edit]", default="edit")
                if selected_mode not in {"run", "edit"}:
                    typer.echo("Invalid mode. Use 'run' or 'edit'.")
                    raise typer.Exit(1)
            _run_notebook(selected_mode, selected_path)
        else:
            typer.echo("TUI canceled. No notebook opened.")


@app.command()
def run(
    input_file: Optional[str] = typer.Option(
        None, "--input-file", "-i", help="Path to .csv (t,y) or .txt thermal step file"
    ),
    dt: Optional[float] = typer.Option(None, "--dt", help="Optional sampling time override"),
    max_taps: Optional[int] = typer.Option(None, "--max-taps", help="Maximum FIR taps"),
):
    """Bypass the TUI and run the detector notebook headlessly."""
    if input_file is None:
        input_file_path = DEFAULT_INPUT_CSV
    else:
        input_file_path = Path(input_file)

    if not input_file_path.exists():
        typer.echo(f"Input file not found: {input_file_path}")
        raise typer.Exit(1)

    _write_defaults(input_file_path, dt, max_taps)
    _show_current_config(_read_defaults(), NOTEBOOK_NAME)
    _run_notebook("run", NOTEBOOK_PATH)


@app.command()
def edit(
    input_file: Optional[str] = typer.Option(
        None, "--input-file", "-i", help="Path to .csv (t,y) or .txt thermal step file"
    ),
    dt: Optional[float] = typer.Option(None, "--dt", help="Optional sampling time override"),
    max_taps: Optional[int] = typer.Option(None, "--max-taps", help="Maximum FIR taps"),
):
    """Bypass the TUI and open the detector notebook in marimo."""
    if input_file is None:
        input_file_path = DEFAULT_INPUT_CSV
    else:
        input_file_path = Path(input_file)

    if not input_file_path.exists():
        typer.echo(f"Input file not found: {input_file_path}")
        raise typer.Exit(1)

    _write_defaults(input_file_path, dt, max_taps)
    _show_current_config(_read_defaults(), NOTEBOOK_NAME)
    _run_notebook("edit", NOTEBOOK_PATH)


if __name__ == "__main__":
    app()
