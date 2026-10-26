"""The ``aeris dashboard`` command: start the Streamlit app."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from typing import Annotated

import typer

from aeris.cli._common import DEFAULT_ROOT, cli_errors


def dashboard(
    root: Annotated[Path, typer.Option("--root", "-r", help="Runs directory.")] = DEFAULT_ROOT,
    dashboards: Annotated[
        Path, typer.Option(help="Directory of saved dashboard layouts (JSON).")
    ] = Path("dashboards"),
    port: Annotated[int, typer.Option(help="Port to serve on.")] = 8501,
) -> None:
    """Open the interactive run dashboard in the browser (needs 'dashboard')."""
    from aeris.dashboard import APP_PATH

    with cli_errors():
        for module in ("streamlit", "plotly"):
            if importlib.util.find_spec(module) is None:
                raise ModuleNotFoundError(f"No module named '{module}'", name=module)
    env = {
        **os.environ,
        "AERIS_RUNS_ROOT": str(root.resolve()),
        "AERIS_DASHBOARDS": str(dashboards.resolve()),
    }
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(APP_PATH),
        "--server.port",
        str(port),
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
    ]
    typer.echo(f"AERIS dashboard on http://localhost:{port}  (runs: {root}, layouts: {dashboards})")
    raise typer.Exit(subprocess.call(command, env=env))
