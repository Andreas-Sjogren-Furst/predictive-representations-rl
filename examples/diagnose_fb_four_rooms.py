"""Diagnose a trusted local AERIS Four Rooms FB checkpoint, no retraining.

Run: uv run --extra fb-torch python examples/diagnose_fb_four_rooms.py runs/fb_four_rooms
"""
from __future__ import annotations
import argparse
from aeris.analysis.fb_checkpoint import analyze_fb_checkpoint


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run_dir', help='completed Four Rooms oracle-data run directory')
    p.add_argument('--plots', action='store_true', help='render an action-alignment map (requires matplotlib)')
    args = p.parse_args()
    report = analyze_fb_checkpoint(args.run_dir)
    for name in (
        'gradient_updates', 'backward_effective_rank',
        'target_policy_entropy_fraction_of_uniform',
        'exact_occupancy_correlation', 'goal_action_advantage_correlation',
        'goal_action_agreement_with_exact_policy_improvement',
    ):
        print(f'{name}: {report[name]}')
    print('first_hit_goal_performance:', report['first_hit_goal_performance'])
    print('Saved analysis/fb_checkpoint_v1.json')
    if args.plots:
        from pathlib import Path
        from aeris.visualization.fb_checkpoint import render_fb_action_alignment
        plot = render_fb_action_alignment(report, Path(args.run_dir) / 'figures/fb_action_alignment.png')
        print('Rendered:', plot)


if __name__ == '__main__':
    main()
