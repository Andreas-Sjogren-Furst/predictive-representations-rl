import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from aeris.analysis.four_rooms import (
    analyze_saved_four_rooms_run,
    build_four_rooms_oracle,
    load_four_rooms_run,
)
from aeris.domain import PolicySnapshot, TabularPolicyReport
from aeris.environments import FourRooms

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"


def with_optimal_snapshot(run):
    oracle = build_four_rooms_oracle(FourRooms(), (9, 9))
    actions = np.argmax(oracle.optimal.optimal_actions, axis=1)
    run.write_policy_snapshot("final", PolicySnapshot(states=oracle.positions, actions=actions))
    return run


def test_run_loader_reads_goal_and_horizon(four_rooms_run):
    run = load_four_rooms_run(four_rooms_run().root)
    assert run.goal == (9, 9)
    assert run.max_steps == 10
    assert run.task_id == "reach_9_9"


def test_run_loader_rejects_incomplete_run(four_rooms_run):
    with pytest.raises(ValueError, match="completed"):
        load_four_rooms_run(four_rooms_run(status="running").root)


def test_offline_analysis_without_matplotlib(four_rooms_run):
    run = with_optimal_snapshot(four_rooms_run())
    analysis = analyze_saved_four_rooms_run(run)
    report = analysis.report
    assert report.snapshots["final"].optimal_action_fraction == 1.0
    assert report.snapshots["final"].success_gap_to_optimal == pytest.approx(0)
    assert 0 < report.optimal_policy.rollout_success_fraction < 1
    assert run.read_report(TabularPolicyReport) == report
    saved = json.loads((run.root / "analysis/tabular_policy_v1.json").read_text())
    assert saved["snapshots"]["final"]["policy_snapshot"] == "policy_final.npz"
    assert set(analysis.snapshots) == {"final"}


def test_offline_analysis_creates_plots(four_rooms_run):
    pytest.importorskip("matplotlib")
    from aeris.visualization.policy_diagnostics import render_tabular_policy_figures

    run = with_optimal_snapshot(four_rooms_run())
    made = render_tabular_policy_figures(analyze_saved_four_rooms_run(run), run.figures_dir)
    assert [p.name for p in made] == ["optimal_policy.png", "policy_gap_final.png"]
    assert all(p.stat().st_size > 1000 for p in made)


def test_cli_analyzes_saved_run(four_rooms_run):
    """The script is a thin, usable CLI rather than the owner of analysis."""
    run = with_optimal_snapshot(four_rooms_run())
    completed = subprocess.run(
        [
            sys.executable,
            str(EXAMPLES / "four_rooms_report.py"),
            "analyze",
            str(run.root),
            "--no-plots",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    assert "final:" in completed.stdout
    assert "tabular_policy_v1.json" in completed.stdout
