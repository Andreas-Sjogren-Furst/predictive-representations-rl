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
from aeris.environments import FourRooms

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"


def make_run(tmp_path, *, status="completed"):
    env = FourRooms()
    oracle = build_four_rooms_oracle(env, (9, 9))
    actions = np.argmax(oracle.optimal.optimal_actions, axis=1)
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": "test",
                "status": status,
                "experiment": {
                    "environment": {"name": "four_rooms"},
                    "evaluation": {
                        "protocol": {"settings": {"max_steps": 10}},
                        "targets": [
                            {"task": {"task_id": "g", "settings": {"goal_row": 9, "goal_col": 9}}}
                        ],
                    },
                },
            }
        )
    )
    np.savez_compressed(tmp_path / "policy_final.npz", positions=oracle.positions, actions=actions)


def test_run_loader_reads_goal_and_horizon(tmp_path):
    make_run(tmp_path)
    run = load_four_rooms_run(tmp_path)
    assert run.goal == (9, 9)
    assert run.max_steps == 10
    assert run.task_id == "g"


def test_run_loader_rejects_incomplete_run(tmp_path):
    make_run(tmp_path, status="running")
    with pytest.raises(ValueError, match="completed"):
        load_four_rooms_run(tmp_path)


def test_offline_analysis_without_matplotlib(tmp_path):
    make_run(tmp_path)
    analysis = analyze_saved_four_rooms_run(tmp_path)
    report = analysis.report
    assert report["snapshots"]["final"]["optimal_action_fraction"] == 1.0
    assert report["snapshots"]["final"]["success_gap_to_optimal"] == pytest.approx(0)
    assert 0 < report["optimal_policy"]["rollout_success_fraction"] < 1
    assert report == json.loads((tmp_path / "analysis/tabular_policy_v1.json").read_text())
    assert set(analysis.snapshots) == {"final"}


def test_offline_analysis_creates_plots(tmp_path):
    pytest.importorskip("matplotlib")
    from aeris.visualization.policy_diagnostics import render_tabular_policy_figures

    make_run(tmp_path)
    made = render_tabular_policy_figures(
        analyze_saved_four_rooms_run(tmp_path), tmp_path / "figures"
    )
    assert [p.name for p in made] == ["optimal_policy.png", "policy_gap_final.png"]
    assert all(p.stat().st_size > 1000 for p in made)


def test_cli_analyzes_saved_run(tmp_path):
    """The script is a thin, usable CLI rather than the owner of analysis."""
    make_run(tmp_path)
    completed = subprocess.run(
        [
            sys.executable,
            str(EXAMPLES / "four_rooms_report.py"),
            "analyze",
            str(tmp_path),
            "--no-plots",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    assert "final:" in completed.stdout
    assert "tabular_policy_v1.json" in completed.stdout
    assert (tmp_path / "analysis" / "tabular_policy_v1.json").exists()
