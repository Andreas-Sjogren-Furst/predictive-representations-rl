"""Inspect a completed Four Rooms run without retraining.

Commands:
  render    Plot recorded artifacts (training curves, evaluations, policies, trajectories).
  analyze   Compare saved policy snapshots with the exact optimal policy.
  diagnose  Validate a trusted FB checkpoint against exact occupancies (needs torch).

Examples:
  uv run --extra viz python examples/four_rooms_report.py render runs/fb_four_rooms
  uv run python examples/four_rooms_report.py analyze runs/fb_four_rooms --no-plots
  uv run --extra fb-torch --extra viz python examples/four_rooms_report.py \
      diagnose runs/fb_four_rooms --plots
"""

from __future__ import annotations

import argparse
from pathlib import Path


def render(args: argparse.Namespace) -> None:
    from aeris.workflows.four_rooms import render_run_figures

    for path in render_run_figures(args.run_dir):
        print(path)


def analyze(args: argparse.Namespace) -> None:
    from aeris.workflows.four_rooms import analyze_saved_policies, render_policy_figures

    analysis = analyze_saved_policies(args.run_dir, gamma=args.gamma)
    for phase, metrics in analysis.report.snapshots.items():
        print(f"{phase}: {metrics}")
    print(f"Saved {args.run_dir / 'analysis' / analysis.report.ARTIFACT_NAME}")
    if args.plots:
        for path in render_policy_figures(args.run_dir, analysis):
            print("Rendered:", path)


def diagnose(args: argparse.Namespace) -> None:
    from aeris.workflows.four_rooms import diagnose_fb_checkpoint, render_fb_figures

    report = diagnose_fb_checkpoint(args.run_dir)
    for name in (
        "gradient_updates",
        "backward_effective_rank",
        "target_policy_entropy_fraction_of_uniform",
        "exact_occupancy_correlation",
        "goal_action_advantage_correlation",
        "goal_action_agreement_with_exact_policy_improvement",
        "final_snapshot_matches_checkpoint",
    ):
        print(f"{name}: {getattr(report, name)}")
    print("first_hit_goal_performance:", report.first_hit_goal_performance)
    print(f"Saved {args.run_dir / 'analysis' / report.ARTIFACT_NAME}")
    if args.plots:
        for path in render_fb_figures(args.run_dir, report):
            print("Rendered:", path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)

    render_parser = commands.add_parser("render", help="plot recorded run artifacts")
    render_parser.set_defaults(handler=render)

    analyze_parser = commands.add_parser("analyze", help="exact tabular policy analysis")
    analyze_parser.add_argument("--gamma", type=float, default=0.99)
    analyze_parser.add_argument(
        "--no-plots",
        dest="plots",
        action="store_false",
        help="compute JSON only, no matplotlib needed",
    )
    analyze_parser.set_defaults(handler=analyze)

    diagnose_parser = commands.add_parser(
        "diagnose", help="FB checkpoint diagnostics (oracle dataset runs)"
    )
    diagnose_parser.add_argument(
        "--plots", action="store_true", help="render an action-alignment map (requires matplotlib)"
    )
    diagnose_parser.set_defaults(handler=diagnose)

    for sub in (render_parser, analyze_parser, diagnose_parser):
        sub.add_argument("run_dir", type=Path, help="completed Four Rooms run directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
