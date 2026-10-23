"""Analyze recorded Four Rooms runs without loading an agent or retraining.

The numerical analysis is reusable without a plotting backend. Plots are
optional derived artifacts produced only when requested.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from aeris.analysis.four_rooms import (
    analyze_saved_four_rooms_run,
    build_four_rooms_oracle,
    compare_four_rooms_snapshot,
)
from aeris.environments import FourRooms


def analyze_run(
    run_dir: str | Path, *, gamma: float = 0.99, make_plots: bool = True
) -> dict[str, Any]:
    """Analyze a saved Four Rooms run and optionally render diagnostic figures.

    Read only recorded experiment artifacts. No checkpoint is loaded, and the
    analysis does not modify the agent or rerun evaluation episodes.

    Args:
        run_dir: Directory containing a completed Four Rooms run.
        gamma: Discount factor used for the exact tabular value baseline.
        make_plots: Write PNG diagnostics in the run's ``figures/`` folder.
            When False, matplotlib need not be installed.

    Returns:
        The numerical report written to ``analysis/tabular_policy_v1.json``.
    """
    root = Path(run_dir)
    report = analyze_saved_four_rooms_run(root, gamma=gamma)

    if not make_plots:
        return report

    # Import visualization only when requested: numerical analysis can run on
    # a headless cluster without a plotting dependency.
    from aeris.visualization.policy_diagnostics import (
        render_optimal_policy,
        render_policy_gap,
    )

    oracle = build_four_rooms_oracle(FourRooms(), tuple(report["goal"]), gamma=gamma)
    figures = root / "figures"
    render_optimal_policy(oracle, figures / "optimal_policy.png")
    for phase, info in report["snapshots"].items():
        actions, diagnostics = compare_four_rooms_snapshot(
            root / info["policy_snapshot"],
            oracle,
            max_steps=report["rollout_horizon"],
        )
        render_policy_gap(
            oracle,
            actions,
            diagnostics,
            figures / f"policy_gap_{phase}.png",
            label=phase.capitalize(),
        )
    return report
