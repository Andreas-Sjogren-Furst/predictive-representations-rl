"""End-to-end: train with the Four Rooms example, then inspect the run with the aeris CLI."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"


def run(*args):
    return subprocess.run(
        [sys.executable, *map(str, args)], check=True, text=True, capture_output=True
    )


def test_train_then_report(tmp_path):
    out = tmp_path / "run"
    run(
        EXAMPLES / "fb_four_rooms.py",
        "--output",
        out,
        "--updates",
        "5",
        "--episodes",
        "2",
        "--max-steps",
        "10",
        "--trajectories",
        "1",
    )
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["status"] == "completed"
    for name in (
        "checkpoint.pt",
        "dataset.npz",
        "policy_initial.npz",
        "policy_final.npz",
        "training_metrics.jsonl",
        "evaluations.jsonl",
        "trajectories.jsonl",
    ):
        assert (out / name).is_file(), name

    aeris = ["-m", "aeris.cli.main"]  # run() prepends the interpreter
    run(*aeris, "analysis", "four-rooms", out, "--no-plots")
    diagnosis = run(*aeris, "analysis", "fb-checkpoint", out, "--no-plots").stdout
    assert "warning" not in diagnosis  # the saved final snapshot matches the checkpoint
    assert (out / "analysis" / "tabular_policy_v1.json").is_file()
    assert (out / "analysis" / "fb_checkpoint_v1.json").is_file()

    pytest.importorskip("matplotlib")
    rendered = run(*aeris, "runs", "plot", out).stdout.split()
    assert {Path(p).name for p in rendered} >= {
        "training.png",
        "policy_final.png",
        "trajectory_final.png",
    }


def test_refuses_nonempty_output(tmp_path):
    (tmp_path / "existing").write_text("x")
    completed = subprocess.run(
        [sys.executable, str(EXAMPLES / "fb_four_rooms.py"), "--output", str(tmp_path)],
        text=True,
        capture_output=True,
    )
    assert completed.returncode != 0
    assert "must be empty" in completed.stderr
