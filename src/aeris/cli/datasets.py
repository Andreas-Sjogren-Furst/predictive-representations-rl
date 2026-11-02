"""Inspect the exact dataset scenarios training can use."""

from __future__ import annotations

import typer

app = typer.Typer(help="Exact (oracle) dataset scenarios.", no_args_is_help=True)


@app.command("list")
def list_scenarios() -> None:
    """Every scenario with its coverage: start states, state-action pairs, transitions."""
    from aeris.workflows.four_rooms_training import scenario_coverage

    typer.echo(f"{'scenario':24} {'states':>7} {'pairs':>7} {'transitions':>12}  description")
    for row in scenario_coverage():
        typer.echo(
            f"{row.name:24} {row.start_states:>3}/{row.all_states} "
            f"{row.pairs:>3}/{row.all_pairs} {row.transitions:>12}  {row.description}"
        )
    typer.echo("Train with: aeris train four-rooms --dataset oracle --scenario NAME")
