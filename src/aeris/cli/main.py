"""AERIS CLI entry point."""

from __future__ import annotations

import typer

from aeris.cli import analysis, runs, train

app = typer.Typer(
    name="aeris",
    help="AERIS — Agent Experimentation and Representation Inspection System.",
    no_args_is_help=True,
)
app.add_typer(train.app, name="train")
app.add_typer(runs.app, name="runs")
app.add_typer(analysis.app, name="analysis")


if __name__ == "__main__":
    app()
