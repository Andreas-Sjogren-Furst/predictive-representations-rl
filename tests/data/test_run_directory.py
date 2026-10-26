import json

import numpy as np
import pytest

from aeris.contracts import (
    ArtifactStore,
    RunReader,
    RunRecorder,
    TrajectoryRecorder,
    TransitionSource,
)
from aeris.data import RunDirectory, TransitionFile, read_policy_snapshot_npz, write_transitions_npz
from aeris.domain import (
    EvaluationRecord,
    EvaluationTrajectory,
    MetricValue,
    PolicySnapshot,
    TabularPolicyReport,
    TrainingMetricRecord,
    TrajectoryStep,
)
from aeris.environments import FourRooms

METRIC = MetricValue("success_rate", 0.5, "fraction", "mean")


def test_run_directory_implements_every_storage_contract(tmp_path):
    run = RunDirectory(tmp_path)
    for contract in (RunRecorder, TrajectoryRecorder, RunReader, ArtifactStore):
        assert isinstance(run, contract)


def test_records_roundtrip_as_typed_objects(four_rooms_run):
    run = four_rooms_run()
    training = TrainingMetricRecord("run-42", "gradient_updates", 5, (METRIC,))
    evaluation = EvaluationRecord(
        "run-42", "checkpoint.pt", "reach_9_9", "goal_reaching_v1", 1, 3, (METRIC,)
    )
    trajectory = EvaluationTrajectory(
        run_id="run-42",
        task_id="t",
        checkpoint_id="c",
        protocol_id="p",
        seed=1,
        episode_index=0,
        initial_observation=np.array([1.0, 0.0]),
        steps=(
            TrajectoryStep(np.array([1.0, 0.0]), 1, np.array([0.0, 1.0]), None, False, False, True),
        ),
        success=True,
    )
    run.write_training_metrics(training)
    run.write_evaluation(evaluation)
    run.write_trajectory(trajectory)
    assert run.read_manifest().run_id == "run-42"
    assert list(run.read_training_metrics()) == [training]
    assert list(run.read_evaluations()) == [evaluation]
    (restored,) = run.read_trajectories()
    assert restored.steps[0].next_observation == [0.0, 1.0]  # arrays come back as JSON lists


def test_missing_streams_and_artifacts_are_empty(tmp_path):
    run = RunDirectory(tmp_path)
    assert not run.exists()
    assert list(run.read_evaluations()) == []
    assert run.read_policy_snapshot("final") is None
    assert run.read_report(TabularPolicyReport) is None
    with pytest.raises(FileNotFoundError, match="no manifest.json"):
        run.read_manifest()


def test_errors_name_the_file_and_line(four_rooms_run):
    run = four_rooms_run()
    path = run.root / "evaluations.jsonl"
    path.write_text("not json\n")
    with pytest.raises(ValueError, match=r"evaluations.jsonl:1"):
        list(run.read_evaluations())
    path.write_text(json.dumps({"run_id": "r"}) + "\n")
    with pytest.raises(ValueError, match=r"evaluations.jsonl:1: EvaluationRecord: missing field"):
        list(run.read_evaluations())


def test_policy_snapshot_npz_keeps_the_existing_format(tmp_path):
    run = RunDirectory(tmp_path)
    snapshot = PolicySnapshot(states=[(1, 1), (1, 2)], actions=[3, 0])
    run.write_policy_snapshot("final", snapshot)
    with np.load(tmp_path / "policy_final.npz") as data:
        assert data["positions"].tolist() == [[1, 1], [1, 2]]
        assert data["actions"].dtype == np.int64
    assert run.read_policy_snapshot("final") == snapshot


def test_malformed_snapshot_file_is_rejected(tmp_path):
    path = tmp_path / "policy_final.npz"
    np.savez_compressed(path, positions=np.zeros((2, 2)), actions=np.zeros(2, dtype=np.int64))
    with pytest.raises(ValueError, match="integer array"):
        read_policy_snapshot_npz(path)


def test_transition_file_verifies_fingerprint_and_yields_transitions(tmp_path):
    env = FourRooms()
    transitions = tuple(env.all_transitions())
    path = tmp_path / "dataset.npz"
    fingerprint = write_transitions_npz(path, transitions)
    dataset = TransitionFile(path, expected_sha256=fingerprint)
    assert isinstance(dataset, TransitionSource)
    assert len(dataset) == len(transitions)
    first = next(dataset.iter_transitions())
    assert first.action == transitions[0].action and first.reward is None
    with pytest.raises(ValueError, match="fingerprint"):
        TransitionFile(path, expected_sha256="0" * 64)


def test_run_catalog_returns_typed_records(four_rooms_run, tmp_path):
    from aeris.data import RunDetails, RunSummary, inspect_run, list_runs, resolve_run

    run = four_rooms_run()
    run.write_training_metrics(TrainingMetricRecord("run-42", "gradient_updates", 5, (METRIC,)))
    (summary,) = list_runs(tmp_path)
    assert isinstance(summary, RunSummary)
    assert (summary.run_id, summary.status, summary.agent) == ("run-42", "completed", "discrete_fb")
    details = inspect_run(resolve_run("run-42", root=tmp_path))
    assert isinstance(details, RunDetails)
    assert details.last_training_record.step == 5
    assert details.recent_evaluations == ()
    assert details.to_dict()["manifest"]["run_id"] == "run-42"
