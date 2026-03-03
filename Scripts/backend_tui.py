from __future__ import annotations
import typer
import subprocess
from pathlib import Path
from typing import Optional

app = typer.Typer()


# Dynamically list available projects (notebooks)
NOTEBOOKS_DIR = Path(__file__).parent.parent / "notebooks"


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


@app.command()
def launch(
    project: Optional[str] = typer.Option(None, "--project", "-p", help="Project to open"),
    backend: Optional[str] = typer.Option(None, "--backend", "-b", help="Simulation backend"),
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
        typer.echo(f"\n🔧 Available backends:")
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
    backend_script = None
    for bname, bscript in BACKENDS:
        if bname == backend:
            backend_script = bscript
            break

    # Open the notebook (marimo)
    project_path = NOTEBOOKS_DIR / project
    typer.echo(f"\n✅ Opening project: {project}")
    subprocess.Popen(["uv", "run", "marimo", "edit", str(project_path)])

    # Run the backend script
    script_path = Path(__file__).parent / backend_script
    typer.echo(f"✅ Running backend: {backend}\n")
    subprocess.run(["powershell", "-ExecutionPolicy", "Bypass", "-File", str(script_path)])


if __name__ == "__main__":
    app()
