"""CLI entry points for existing analyses (not new analysis implementations)."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(help="Analyze saved experiment artifacts.", no_args_is_help=True)


@app.command("four-rooms")
def four_rooms(
    run_dir: Annotated[Path, typer.Argument(help="Saved Four Rooms run directory.")],
    gamma: Annotated[
        float, typer.Option("--gamma", help="Discount for the exact tabular baseline.")
    ] = 0.99,
    no_plots: Annotated[
        bool, typer.Option("--no-plots", help="Compute JSON without Matplotlib.")
    ] = False,
) -> None:
    """Analyze saved Four Rooms policies; no checkpoint loading or training."""
    # Keep optional numerical/graphics dependencies out of `aeris --help`.
    from aeris.analysis.four_rooms import analyze_saved_four_rooms_run

    try:
        analysis = analyze_saved_four_rooms_run(run_dir, gamma=gamma)
        if not no_plots:
            from aeris.visualization.policy_diagnostics import render_tabular_policy_figures

            render_tabular_policy_figures(analysis, run_dir / "figures")
    except (OSError, ValueError, KeyError, ImportError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    report = analysis.report
    typer.echo(f"Analyzed run: {report.run_id}")
    for phase, metrics in report.snapshots.items():
        typer.echo(
            f"  {phase}: optimal action fraction={metrics.optimal_action_fraction:.1%}, "
            f"goal success={metrics.rollout_success_fraction:.1%}"
        )
    typer.echo(f"Report: {run_dir / 'analysis' / report.ARTIFACT_NAME}")
    if not no_plots:
        typer.echo(f"Figures: {run_dir / 'figures'}")
