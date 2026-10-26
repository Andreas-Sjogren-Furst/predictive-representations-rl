"""Arguments and error handling shared by the CLI commands."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated

import typer

RunArg = Annotated[
    str, typer.Argument(help="Run directory, folder name under --root, or unique run_id.")
]
RootOpt = Annotated[
    Path, typer.Option("--root", "-r", help="Runs directory for name/run_id lookup.")
]
NoPlotsOpt = Annotated[
    bool, typer.Option("--no-plots", help="Skip figures (no Matplotlib needed).")
]
JsonOpt = Annotated[bool, typer.Option("--json", help="Emit machine-readable JSON.")]
DEFAULT_ROOT = Path("runs")

# Optional dependencies and the extra that provides them.
_EXTRAS = {
    "torch": "fb-torch",
    "matplotlib": "viz",
    "streamlit": "dashboard",
    "plotly": "dashboard",
}


@contextmanager
def cli_errors() -> Iterator[None]:
    """Turn expected failures into a one-line message and exit code 1."""
    try:
        yield
    except ModuleNotFoundError as exc:
        extra = _EXTRAS.get((exc.name or "").split(".")[0])
        hint = f" (install the '{extra}' extra: uv sync --extra {extra})" if extra else ""
        typer.echo(f"Error: {exc}{hint}", err=True)
        raise typer.Exit(code=1) from exc
    except (OSError, ValueError, RuntimeError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
