"""Commands for discovering and inspecting saved runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from aeris.cli._common import DEFAULT_ROOT, JsonOpt, RootOpt, RunArg, cli_errors
from aeris.data.run_catalog import inspect_run, list_runs, resolve_run
from aeris.domain.results import EvaluationRecord, TrainingMetricRecord

app = typer.Typer(help="Browse and plot existing runs.", no_args_is_help=True)


def _metrics(record: TrainingMetricRecord | EvaluationRecord) -> str:
    return ", ".join(f"{m.name}={m.value}" for m in record.metrics) or "none"


@app.command("list")
def list_command(
    root: Annotated[Path, typer.Option("--root", "-r", help="Runs directory.")] = DEFAULT_ROOT,
    as_json: JsonOpt = False,
) -> None:
    """List immediate run directories containing manifest.json."""
    with cli_errors():
        records = list_runs(root)
    if as_json:
        typer.echo(json.dumps([record.to_dict() for record in records], indent=2))
        return
    if not records:
        typer.echo(f"No saved runs found in {root}")
        return
    typer.echo(f"{'STATUS':<12} {'RUN ID':<30} {'AGENT':<16} DIRECTORY")
    for record in records:
        typer.echo(f"{record.status:<12} {record.run_id:<30} {record.agent:<16} {record.directory}")


@app.command("show")
def show_command(
    run: RunArg,
    root: RootOpt = DEFAULT_ROOT,
    as_json: JsonOpt = False,
) -> None:
    """Show manifest information and the most recent recorded metrics."""
    with cli_errors():
        details = inspect_run(resolve_run(run, root=root))
    if as_json:
        typer.echo(json.dumps(details.to_dict(), indent=2))
        return
    manifest = details.manifest
    dataset = manifest.experiment.dataset
    typer.echo(f"Run:        {details.run_id}")
    typer.echo(f"Directory:  {details.directory}")
    typer.echo(f"Status:     {details.status}")
    typer.echo(f"Experiment: {details.experiment}")
    typer.echo(f"Agent:      {details.agent}")
    typer.echo(f"Seed:       {details.seed}")
    typer.echo(f"Dataset:    {dataset.name if dataset is not None else 'none'}")
    typer.echo(f"Checkpoint: {manifest.selected_checkpoint or 'none'}")
    typer.echo(f"Training records: {details.training_record_count}")
    last = details.last_training_record
    if last is not None:
        typer.echo(f"  Last {last.step_axis} {last.step}: {_metrics(last)}")
    typer.echo(f"Evaluation records: {details.evaluation_record_count}")
    for record in details.recent_evaluations:
        typer.echo(f"  {record.checkpoint_id} / {record.task_id}: {_metrics(record)}")
    if manifest.error:
        typer.echo(f"Error: {manifest.error}")


@app.command("plot")
def plot_command(run: RunArg, root: RootOpt = DEFAULT_ROOT) -> None:
    """Plot a Four Rooms run's recorded artifacts into its figures/ folder (needs 'viz')."""
    from aeris.workflows.four_rooms import render_run_figures

    with cli_errors():
        made = render_run_figures(resolve_run(run, root=root))
    if not made:
        typer.echo("Nothing to plot: the run has no recorded metrics, snapshots or trajectories")
    for path in made:
        typer.echo(f"Figure: {path}")


@app.command("summary")
def summary_command(
    root: Annotated[Path, typer.Option("--root", "-r", help="Runs directory.")] = DEFAULT_ROOT,
    group_by: Annotated[
        str,
        typer.Option(
            "--group-by",
            help="Recorded value to group by: name, seed, dataset.origin, agent.<key>, …",
        ),
    ] = "name",
    metric: Annotated[
        list[str] | None, typer.Option("--metric", "-m", help="Only metrics containing this text.")
    ] = None,
    as_json: JsonOpt = False,
) -> None:
    """Mean ± std across seeds of every completed run's headline numbers, per group."""
    from aeris.workflows.summary import summarize_runs

    with cli_errors():
        groups = summarize_runs(root, by=group_by)
    selected = [
        (
            group,
            {k: v for k, v in group.metrics.items() if not metric or any(m in k for m in metric)},
        )
        for group in groups
    ]
    if as_json:
        data = [
            {
                "group": group.group,
                "runs": list(group.runs),
                "seeds": list(group.seeds),
                "metrics": {
                    k: {"mean": s.mean, "std": s.std, "n": s.n} for k, s in metrics.items()
                },
            }
            for group, metrics in selected
        ]
        typer.echo(json.dumps(data, indent=2))
        return
    if not groups:
        typer.echo(f"No completed runs in {root}")
        return
    for group, metrics in selected:
        seeds = ", ".join(map(str, group.seeds))
        typer.echo(f"{group_by} = {group.group}  ({len(group.runs)} runs, seeds {seeds})")
        width = max((len(k) for k in metrics), default=0)
        for name, stat in metrics.items():
            typer.echo(f"  {name:<{width}}  {stat.mean:7.3f} ± {stat.std:.3f}  (n={stat.n})")
        typer.echo("")
