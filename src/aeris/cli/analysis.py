"""CLI entry points for the analysis workflows (no analysis is implemented here)."""

from __future__ import annotations

from typing import Annotated

import typer

from aeris.cli._common import DEFAULT_ROOT, NoPlotsOpt, RootOpt, RunArg, cli_errors
from aeris.data.run_catalog import resolve_run

app = typer.Typer(
    help="Analyze saved runs; results are saved under analysis/.", no_args_is_help=True
)


def _echo_paths(label: str, paths) -> None:
    for path in paths:
        typer.echo(f"{label}: {path}")


@app.command("four-rooms")
def four_rooms(
    run: RunArg,
    root: RootOpt = DEFAULT_ROOT,
    gamma: Annotated[
        float, typer.Option("--gamma", help="Discount for the exact tabular baseline.")
    ] = 0.99,
    no_plots: NoPlotsOpt = False,
) -> None:
    """Compare saved Four Rooms policy snapshots with the exact optimal policy (no torch)."""
    # Workflows are imported lazily to keep optional dependencies out of `aeris --help`.
    from aeris.workflows.four_rooms import analyze_saved_policies, render_policy_figures

    with cli_errors():
        store = resolve_run(run, root=root)
        analysis = analyze_saved_policies(store, gamma=gamma)
        figures = () if no_plots else render_policy_figures(store, analysis)
    report = analysis.report
    typer.echo(f"Analyzed run: {report.run_id}")
    for phase, metrics in report.snapshots.items():
        typer.echo(
            f"  {phase}: optimal action fraction={metrics.optimal_action_fraction:.1%}, "
            f"goal success={metrics.rollout_success_fraction:.1%}"
        )
    typer.echo(f"Report: {store.report_path(type(report))}")
    _echo_paths("Figure", figures)


@app.command("fb-checkpoint")
def fb_checkpoint(run: RunArg, root: RootOpt = DEFAULT_ROOT, no_plots: NoPlotsOpt = False) -> None:
    """Check a trusted FB checkpoint against exact occupancies (needs 'fb-torch')."""
    from aeris.workflows.four_rooms import diagnose_fb_checkpoint, render_fb_figures

    with cli_errors():
        store = resolve_run(run, root=root)
        report = diagnose_fb_checkpoint(store)
        figures = () if no_plots else render_fb_figures(store, report)
    typer.echo(f"Diagnosed checkpoint: {report.run_id} / {report.checkpoint}")
    typer.echo(f"  gradient updates: {report.gradient_updates}")
    typer.echo(f"  backward effective rank: {report.backward_effective_rank:.3f}")
    typer.echo(f"  exact occupancy correlation: {_maybe(report.exact_occupancy_correlation)}")
    typer.echo(
        "  agreement with exact policy improvement: "
        f"{report.goal_action_agreement_with_exact_policy_improvement:.1%}"
    )
    first_hit = report.first_hit_goal_performance
    typer.echo(f"  first-hit goal success: {first_hit.rollout_success_fraction:.1%}")
    if report.final_snapshot_matches_checkpoint is False:
        typer.echo("  warning: the saved final policy snapshot does not match this checkpoint")
    typer.echo(f"Report: {store.report_path(type(report))}")
    _echo_paths("Figure", figures)


def _maybe(value: float | None) -> str:
    return "undefined (no variation)" if value is None else f"{value:.3f}"


def _parse_goal(text: str) -> tuple[int, int]:
    try:
        row, col = (int(part) for part in text.split(","))
    except ValueError as exc:
        raise typer.BadParameter(f"expected ROW,COL such as 9,9, got {text!r}") from exc
    return row, col


@app.command("zero-shot")
def zero_shot(
    run: RunArg,
    root: RootOpt = DEFAULT_ROOT,
    suite: Annotated[str, typer.Option(help="Task suite to evaluate on.")] = "four_rooms_v1",
    samples: Annotated[
        int | None,
        typer.Option(
            help="Random pretraining observations used as reward evidence (default: all)."
        ),
    ] = None,
    seed: Annotated[int, typer.Option(help="Seed for evidence sampling.")] = 0,
) -> None:
    """Infer every task of a suite and score the frozen policy exactly (needs 'fb-torch')."""
    from aeris.workflows.four_rooms import evaluate_task_suite

    with cli_errors():
        store = resolve_run(run, root=root)
        report = evaluate_task_suite(store, suite=suite, samples=samples, seed=seed)
    protocols = list(report.evidence_protocols)
    typer.echo(f"Zero-shot suite {report.suite}: {report.run_id} / {report.checkpoint}")
    typer.echo("  normalized score (0 = random policy, 1 = optimal), mean per family:")
    typer.echo(f"  {'family':<15}{'tasks':>6}" + "".join(f"{p:>18}" for p in protocols))
    for family, means in report.mean_score.items():
        count = (
            len(report.tasks) if family == "all" else sum(t.family == family for t in report.tasks)
        )
        cells = "".join(f"{_score(means[p]):>18}" for p in protocols)
        typer.echo(f"  {family:<15}{count:>6}{cells}")
    invisible = sum(1 for t in report.tasks if t.outcomes["dataset_rewards"].rewarded_samples == 0)
    if invisible:
        typer.echo(f"  {invisible} tasks had no rewarded sample in the dataset evidence")
    typer.echo(f"Report: {store.report_path(type(report))}")


def _score(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


@app.command("representations")
def representations(
    run: RunArg,
    root: RootOpt = DEFAULT_ROOT,
    goal: Annotated[
        list[str] | None,
        typer.Option("--goal", help="Goal cell ROW,COL; repeat for several (default: every cell)."),
    ] = None,
) -> None:
    """Save FB representations for the dashboard: B, and F/policy/exact density per goal."""
    from aeris.workflows.four_rooms import export_fb_representations

    goals = [_parse_goal(text) for text in goal] if goal else None
    with cli_errors():
        store = resolve_run(run, root=root)
        result = export_fb_representations(store, goals=goals)
    states, d = result.backward.shape
    typer.echo(f"Representations: {result.run_id} / {result.checkpoint}")
    typer.echo(f"  {states} states, d = {d}, {len(result.goals)} goals")
    typer.echo(f"Artifact: {store.arrays_path(result.ARTIFACT_NAME)}")
    typer.echo("View: aeris dashboard (layout 'Representations')")
