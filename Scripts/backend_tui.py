from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer()


# Dynamically list available projects (notebooks)
NOTEBOOKS_DIR = Path(__file__).parent.parent / "notebooks"
INPUT_DIR = Path(__file__).parent.parent / "input" / "impulse_response"
DEFAULTS_PATH = INPUT_DIR / "notebook_defaults.json"
DEFAULT_INPUT_FILE = INPUT_DIR / "sample_impulse.csv"


def list_notebooks():
    if not NOTEBOOKS_DIR.exists():
        return []
    return [
        f.name for f in NOTEBOOKS_DIR.glob("*.py") if f.is_file() and not f.name.startswith("AGENT")
    ]  # skip AGENT.md


# List available backends
BACKENDS = [
    ("control", "run_backend_control.ps1"),
    ("collimator", "run_backend_collimator.ps1"),
]


def _configure_impulse_notebook_defaults(
    input_file: Optional[str],
    dt: Optional[float],
    max_taps: Optional[int],
) -> None:
    INPUT_DIR.mkdir(parents=True, exist_ok=True)

    if input_file is None:
        typed_path = typer.prompt(
            "Path to impulse input file (.csv or .txt)",
            default=str(DEFAULT_INPUT_FILE),
            show_default=True,
        )
        resolved_input = Path(typed_path)
    else:
        resolved_input = Path(input_file)

    if not resolved_input.exists():
        typer.echo(f"Input file not found: {resolved_input}")
        raise typer.Exit(1)

    effective_dt = dt if dt is not None else None
    effective_max_taps = max_taps if max_taps is not None else 300

    defaults = {
        "input_csv": str(resolved_input),
        "dt": effective_dt,
        "max_taps": int(effective_max_taps),
        "output_model_json": str(INPUT_DIR / "identified_model.json"),
    }

    DEFAULTS_PATH.write_text(json.dumps(defaults, indent=2), encoding="utf-8")
    typer.echo(f"Impulse notebook defaults updated: {DEFAULTS_PATH}")


@app.command()
def launch(
    project: Optional[str] = typer.Option(None, "--project", "-p", help="Project to open"),
    backend: Optional[str] = typer.Option(None, "--backend", "-b", help="Simulation backend"),
    input_file: Optional[str] = typer.Option(
        None,
        "--input-file",
        "-i",
        help="Impulse input (.csv with t,y or thermal .txt) used by notebooks/04_impulse_response_analysis.py",
    ),
    dt: Optional[float] = typer.Option(
        None,
        "--dt",
        help="Optional sample time override used by impulse-response notebook",
    ),
    max_taps: Optional[int] = typer.Option(
        None,
        "--max-taps",
        help="Maximum FIR taps used by impulse-response notebook",
    ),
):
    """Select project (notebook) and backend, then launch simulation."""
    notebooks = list_notebooks()
    if not notebooks:
        typer.echo("No projects (notebooks) found in the 'notebooks/' directory.")
        raise typer.Exit(1)

    # Project selection
    if project is None:
        typer.echo("\n📚 Available projects:")
        for idx, nb in enumerate(notebooks, 1):
            typer.echo(f"  {idx}. {nb}")
        while True:
            try:
                selection = int(typer.prompt("\nSelect a project by number"))
                if 1 <= selection <= len(notebooks):
                    project = notebooks[selection - 1]
                    break
                else:
                    typer.echo(
                        f"❌ Invalid selection. Please enter a number between 1 and {len(notebooks)}."
                    )
            except ValueError:
                typer.echo("❌ Please enter a valid number.")
    else:
        if project not in notebooks:
            typer.echo(f"❌ Project '{project}' not found.")
            raise typer.Exit(1)

    # Backend selection
    if backend is None:
        typer.echo("\n🔧 Available backends:")
        for idx, (bname, _) in enumerate(BACKENDS, 1):
            typer.echo(f"  {idx}. {bname}")
        while True:
            try:
                selection = int(typer.prompt("\nSelect a backend by number"))
                if 1 <= selection <= len(BACKENDS):
                    backend = BACKENDS[selection - 1][0]
                    break
                else:
                    typer.echo(
                        f"❌ Invalid selection. Please enter a number between 1 and {len(BACKENDS)}."
                    )
            except ValueError:
                typer.echo("❌ Please enter a valid number.")
    else:
        backend_names = [b[0] for b in BACKENDS]
        if backend not in backend_names:
            typer.echo(f"❌ Unknown backend: {backend}. Choose from: {', '.join(backend_names)}")
            raise typer.Exit(1)

    # Get the backend script
    backend_script = next((bscript for bname, bscript in BACKENDS if bname == backend), None)
    if backend_script is None:
        typer.echo(f"❌ Could not resolve backend script for: {backend}")
        raise typer.Exit(1)

    # Open the notebook (marimo)
    project_path = NOTEBOOKS_DIR / project

    if project == "04_impulse_response_analysis.py":
        _configure_impulse_notebook_defaults(input_file=input_file, dt=dt, max_taps=max_taps)

    typer.echo(f"\n✅ Opening project: {project}")
    subprocess.Popen(["uv", "run", "marimo", "edit", str(project_path)])

    if project == "04_impulse_response_analysis.py":
        typer.echo("✅ Impulse notebook started with updated defaults. No backend required.\n")
        return

    # Run the backend script
    script_path = Path(__file__).parent / backend_script
    typer.echo(f"✅ Running backend: {backend}\n")
    subprocess.run(["powershell", "-ExecutionPolicy", "Bypass", "-File", str(script_path)])


if __name__ == "__main__":
    app()
