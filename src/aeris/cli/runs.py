"""Commands for discovering and inspecting saved runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from aeris.data.run_catalog import inspect_run, list_runs, resolve_run

app = typer.Typer(help="Browse existing run artifacts.", no_args_is_help=True)


def _fail(exc: Exception) -> None:
    typer.echo(f"Error: {exc}", err=True)
    raise typer.Exit(code=1)


def _metrics(record: dict | None) -> str:
    if not record:
        return "none"
    values = record.get("metrics", [])
    return ", ".join(f"{v.get('name', '?')}={v.get('value', '?')}" for v in values) or "none"


@app.command("list")
def list_command(
    root: Annotated[Path, typer.Option("--root", "-r", help="Runs directory.")] = Path("runs"),
    as_json: Annotated[bool, typer.Option("--json", help="Emit machine-readable JSON.")] = False,
) -> None:
    """List immediate run directories containing manifest.json."""
    try:
        records = list_runs(root)
    except (OSError, ValueError) as exc:
        _fail(exc)
    if as_json:
        typer.echo(json.dumps(records, indent=2))
        return
    if not records:
        typer.echo(f"No saved runs found in {root}")
        return
    typer.echo(f"{'STATUS':<12} {'RUN ID':<30} {'AGENT':<16} DIRECTORY")
    for record in records:
        typer.echo(
            f"{str(record['status']):<12} {record['run_id']:<30} "
            f"{record['agent']:<16} {record['directory']}"
        )


@app.command("show")
def show_command(
    run: Annotated[str, typer.Argument(help="Run directory, folder name, or unique run_id.")],
    root: Annotated[Path, typer.Option("--root", "-r", help="Runs directory for lookup.")] = Path(
        "runs"
    ),
    as_json: Annotated[bool, typer.Option("--json", help="Emit machine-readable JSON.")] = False,
) -> None:
    """Show manifest information and the most recent recorded metrics."""
    try:
        details = inspect_run(resolve_run(run, root=root))
    except (OSError, ValueError) as exc:
        _fail(exc)
    if as_json:
        typer.echo(json.dumps(details, indent=2))
        return
    manifest = details["manifest"]
    exp = manifest["experiment"]
    typer.echo(f"Run:        {details['run_id']}")
    typer.echo(f"Directory:  {details['directory']}")
    typer.echo(f"Status:     {details['status']}")
    typer.echo(f"Experiment: {details['experiment']}")
    typer.echo(f"Agent:      {details['agent']}")
    typer.echo(f"Seed:       {details['seed']}")
    dataset = exp.get("dataset")
    typer.echo(
        f"Dataset:    {dataset.get('name', 'unknown') if isinstance(dataset, dict) else 'none'}"
    )
    typer.echo(f"Checkpoint: {manifest.get('selected_checkpoint') or 'none'}")
    typer.echo(f"Training records: {details['training_record_count']}")
    last = details["last_training_record"]
    if last:
        typer.echo(
            f"  Last {last.get('step_axis', 'step')} {last.get('step', '?')}: {_metrics(last)}"
        )
    typer.echo(f"Evaluation records: {details['evaluation_record_count']}")
    for record in details["recent_evaluations"]:
        typer.echo(
            f"  {record.get('checkpoint_id', '?')} / {record.get('task_id', '?')}: "
            f"{_metrics(record)}"
        )
    if manifest.get("error"):
        typer.echo(f"Error: {manifest['error']}")
