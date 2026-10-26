"""The CLI reads actual artifact formats; scientific work stays in AERIS modules."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from typer.testing import CliRunner

from aeris.analysis.four_rooms import build_four_rooms_oracle
from aeris.cli.main import app
from aeris.domain import EvaluationRecord, MetricValue, PolicySnapshot, TrainingMetricRecord
from aeris.environments import FourRooms
from conftest import write_four_rooms_run

runner = CliRunner()


def _run(directory: Path, *, identifier: str = "run-42", environment: str = "four_rooms"):
    """A real run directory: manifest, three training records and one evaluation."""
    run = write_four_rooms_run(directory, run_id=identifier, environment=environment)
    for i in range(3):
        run.write_training_metrics(
            TrainingMetricRecord(
                identifier,
                "gradient_updates",
                i,
                (MetricValue("fb_loss", i * 0.25, "unitless", "instant"),),
            )
        )
    run.write_evaluation(
        EvaluationRecord(
            identifier,
            "checkpoint.pt",
            "goal",
            "goal_reaching_v1",
            42,
            2,
            (MetricValue("success_rate", 0.5, "fraction", "mean"),),
        )
    )
    return run


def test_help_contains_only_implemented_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0, result.output
    assert "runs" in result.output
    assert "analysis" in result.output
    assert "datasets" not in result.output
    assert runner.invoke(app, ["runs", "--help"]).exit_code == 0
    assert runner.invoke(app, ["analysis", "--help"]).exit_code == 0


def test_list_runs_without_loading_checkpoints(tmp_path):
    folder = _run(tmp_path / "arbitrary_directory_name").root
    (folder / "checkpoint.pt").write_bytes(b"not a real checkpoint")
    (tmp_path / "not_a_run").mkdir()
    result = runner.invoke(app, ["runs", "list", "--root", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "run-42" in result.output
    assert "discrete_fb" in result.output
    assert "not_a_run" not in result.output
    structured = runner.invoke(app, ["runs", "list", "--root", str(tmp_path), "--json"])
    assert structured.exit_code == 0
    rows = json.loads(structured.stdout)
    assert len(rows) == 1
    assert rows[0]["directory"] == str(folder)


def test_show_accepts_path_name_and_run_id(tmp_path):
    _run(tmp_path / "folder")
    for ref in (str(tmp_path / "folder"), "folder", "run-42"):
        args = ["runs", "show", ref, "--root", str(tmp_path)]
        result = runner.invoke(app, args)
        assert result.exit_code == 0, result.output
        assert "Training records: 3" in result.output
        assert "fb_loss=0.5" in result.output
        assert "success_rate=0.5" in result.output
    json_result = runner.invoke(app, ["runs", "show", "run-42", "--root", str(tmp_path), "--json"])
    obj = json.loads(json_result.stdout)
    assert obj["manifest"]["selected_checkpoint"] == "checkpoint.pt"
    assert obj["last_training_record"]["step"] == 2


def test_missing_runs_and_ambiguous_run_id_have_helpful_errors(tmp_path):
    missing = runner.invoke(app, ["runs", "list", "--root", str(tmp_path / "missing")])
    assert missing.exit_code != 0
    assert "does not exist" in missing.output
    _run(tmp_path / "a", identifier="duplicate")
    _run(tmp_path / "b", identifier="duplicate")
    duplicate = runner.invoke(app, ["runs", "show", "duplicate", "--root", str(tmp_path)])
    assert duplicate.exit_code != 0
    assert "multiple runs" in duplicate.output


def test_invalid_manifest_is_reported(tmp_path):
    directory = tmp_path / "broken"
    directory.mkdir()
    (directory / "manifest.json").write_text("not JSON", encoding="utf8")
    result = runner.invoke(app, ["runs", "list", "--root", str(tmp_path)])
    assert result.exit_code != 0
    assert "invalid JSON" in result.output


def test_analysis_calls_existing_saved_run_implementation(tmp_path):
    run = _run(tmp_path / "run")
    directory = run.root
    oracle = build_four_rooms_oracle(FourRooms(), (9, 9))
    optimal_actions = np.argmax(oracle.optimal.optimal_actions, axis=1)
    run.write_policy_snapshot(
        "final", PolicySnapshot(states=oracle.positions, actions=optimal_actions)
    )
    result = runner.invoke(app, ["analysis", "four-rooms", str(directory), "--no-plots"])
    assert result.exit_code == 0, result.output
    assert "optimal action fraction=100.0%" in result.output
    report_file = directory / "analysis" / "tabular_policy_v1.json"
    assert report_file.exists()
    assert not (directory / "figures").exists()
    report = json.loads(report_file.read_text(encoding="utf8"))
    assert report["run_id"] == "run-42"
    assert report["snapshots"]["final"]["optimal_action_fraction"] == 1.0


def test_analysis_invalid_environment_exits_cleanly(tmp_path):
    directory = _run(tmp_path / "run", environment="other_env").root
    result = runner.invoke(app, ["analysis", "four-rooms", str(directory), "--no-plots"])
    assert result.exit_code == 1
    assert "only Four Rooms runs are supported" in result.output
