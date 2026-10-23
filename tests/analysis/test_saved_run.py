import json
from pathlib import Path

import numpy as np
import pytest

from aeris.analysis.four_rooms import build_four_rooms_oracle
from aeris.environments import FourRooms


def _analyze(tmp_path, *, with_plots):
    from aeris.analysis.saved_run import analyze_run
    env = FourRooms()
    oracle = build_four_rooms_oracle(env, (9, 9))
    actions = np.argmax(oracle.optimal.optimal_actions, axis=1)
    (tmp_path / "manifest.json").write_text(json.dumps({
        "run_id": "test", "status": "completed", "experiment": {
            "environment": {"name": "four_rooms"},
            "evaluation": {"protocol": {"settings": {"max_steps": 10}},
                           "targets": [{"task": {"task_id": "g", "settings": {"goal_row": 9, "goal_col": 9}}}]},
        }
    }))
    np.savez_compressed(tmp_path / "policy_final.npz", positions=oracle.positions, actions=actions)
    report = analyze_run(tmp_path, make_plots=with_plots)
    assert report["snapshots"]["final"]["optimal_action_fraction"] == 1.0
    assert report["snapshots"]["final"]["success_gap_to_optimal"] == pytest.approx(0)
    assert 0 < report["optimal_policy"]["rollout_success_fraction"] < 1
    assert (tmp_path / "analysis/tabular_policy_v1.json").exists()
    assert report == json.loads((tmp_path / "analysis/tabular_policy_v1.json").read_text())
    if with_plots:
        for name in ("optimal_policy.png", "policy_gap_final.png"):
            assert (tmp_path / "figures" / name).stat().st_size > 1000


def test_offline_analysis_without_matplotlib(tmp_path):
    _analyze(tmp_path, with_plots=False)


def test_offline_analysis_creates_plots(tmp_path):
    pytest.importorskip("matplotlib")
    _analyze(tmp_path, with_plots=True)


def test_cli_analyzes_saved_run(tmp_path):
    """The script is a thin, usable CLI rather than the owner of analysis."""
    import subprocess
    import sys

    _analyze(tmp_path, with_plots=False)
    script = Path(__file__).resolve().parents[2] / "examples" / "analyze_four_rooms.py"
    completed = subprocess.run(
        [sys.executable, str(script), str(tmp_path), "--no-plots"],
        check=True,
        text=True,
        capture_output=True,
    )
    assert "final:" in completed.stdout
    assert "tabular_policy_v1.json" in completed.stdout
    assert (tmp_path / "analysis" / "tabular_policy_v1.json").exists()
