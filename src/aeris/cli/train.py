"""CLI entry points for the training workflows."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer

from aeris.cli._common import cli_errors

app = typer.Typer(help="Train agents and save self-describing runs.", no_args_is_help=True)


class Dataset(StrEnum):
    oracle = "oracle"
    random = "random"


@app.command("four-rooms")
def four_rooms(
    output: Annotated[
        Path, typer.Option("--output", "-o", help="New, empty run directory.")
    ] = Path("runs/fb_four_rooms"),
    dataset: Annotated[
        Dataset,
        typer.Option(help="Full state-action coverage ('oracle') or random collection."),
    ] = Dataset.oracle,
    transitions: Annotated[int, typer.Option(help="Random collection size.")] = 5000,
    updates: Annotated[int, typer.Option(help="Gradient updates.")] = 250,
    batch_size: Annotated[int, typer.Option(help="Transitions per update.")] = 128,
    episodes: Annotated[int, typer.Option(help="Evaluation episodes per checkpoint.")] = 20,
    max_steps: Annotated[int, typer.Option(help="Episode horizon.")] = 60,
    seed: Annotated[int, typer.Option(help="Seed for data, agent and evaluation.")] = 7,
    trajectories: Annotated[
        int, typer.Option(help="Evaluation episodes saved per checkpoint (0 disables).")
    ] = 3,
    goal_row: Annotated[int, typer.Option(help="Evaluation goal row.")] = 9,
    goal_col: Annotated[int, typer.Option(help="Evaluation goal column.")] = 9,
    representation_dim: Annotated[int, typer.Option(help="FB representation size d.")] = 100,
    hidden_dim: Annotated[int, typer.Option(help="Hidden units per FB network layer.")] = 256,
    log_every: Annotated[
        int | None,
        typer.Option(help="Updates between logged losses (default: 100 points per run)."),
    ] = None,
    plots: Annotated[
        bool, typer.Option("--plots", help="Plot the run afterwards ('viz').")
    ] = False,
) -> None:
    """Train discrete FB offline on Four Rooms (needs 'fb-torch'); not a paper reproduction."""
    from aeris.workflows.four_rooms_training import FourRoomsFBTraining, train_four_rooms_fb

    with cli_errors():
        settings = FourRoomsFBTraining(
            output=output,
            dataset=dataset.value,
            transitions=transitions,
            updates=updates,
            batch_size=batch_size,
            episodes=episodes,
            max_steps=max_steps,
            seed=seed,
            trajectories=trajectories,
            goal=(goal_row, goal_col),
            representation_dim=representation_dim,
            hidden_dim=hidden_dim,
            log_every=log_every,
        )
        outcome = train_four_rooms_fb(settings)
    typer.echo(f"Trained run: {settings.run_id}")
    typer.echo(
        f"  transitions: {outcome.transition_count} | gradient updates: {outcome.update_count}"
    )
    for label, record in (("initial", outcome.initial), ("final", outcome.final)):
        metrics = ", ".join(f"{m.name}={m.value:.4g}" for m in record.metrics)
        typer.echo(f"  {label}: {metrics}")
    typer.echo(f"Run: {outcome.run.root}")
    provenance = outcome.run.read_manifest().provenance
    if provenance.get("git_commit") is None:
        typer.echo("  note: not a git checkout, so no code revision was recorded")
    elif provenance.get("git_dirty"):
        typer.echo(
            "  warning: AERIS had uncommitted changes; the recorded commit "
            f"{provenance['git_commit'][:12]} does not fully identify the code"
        )
    # Plotting runs after the manifest is completed: a rendering failure must
    # never mark a successfully trained run as failed.
    if plots:
        from aeris.workflows.four_rooms import render_run_figures

        with cli_errors():
            made = render_run_figures(outcome.run)
        for path in made:
            typer.echo(f"Figure: {path}")
