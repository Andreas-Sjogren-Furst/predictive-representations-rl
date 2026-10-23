import json

import pytest

from aeris.domain import (
    ComponentSpec,
    EvaluationRecord,
    ExperimentSpec,
    MetricValue,
    RunManifest,
    TaskEvidenceRef,
    TrainingPlan,
)


def test_run_manifest_carries_experiment_config_and_provenance():
    spec = ExperimentSpec(
        name="demo",
        agent=ComponentSpec("fb"),
        dataset=ComponentSpec("d1"),
        training=TrainingPlan("offline", ComponentSpec("offline_updates")),
        seed=4,
    )
    run = RunManifest(
        run_id="run-42",
        experiment=spec,
        status="completed",
        selected_checkpoint="cpt-8",
        provenance={"git_commit": "abc123", "dataset": {"sha256": "deadbeef"}},
    )
    exported = json.loads(json.dumps(run.to_dict()))
    assert exported["experiment"]["agent"]["name"] == "fb"
    assert exported["provenance"]["dataset"]["sha256"] == "deadbeef"
    assert exported["selected_checkpoint"] == "cpt-8"


def test_evaluation_record_identifies_protocol_and_evidence():
    record = EvaluationRecord(
        run_id="run-42",
        checkpoint_id="cpt-8",
        task_id="reach_goal",
        protocol_id="zero_shot_episodic",
        seed=42,
        episode_count=10,
        metrics=(MetricValue("return", 4.0, "reward", "mean"),),
        inference_evidence=TaskEvidenceRef("reward_samples", "inference"),
    )
    assert record.to_dict()["metrics"][0]["unit"] == "reward"
    assert record.inference_evidence.source_id == "reward_samples"
    assert record.to_dict()["inference_evidence"]["split"] == "inference"


def test_evaluation_rejects_ambiguous_duplicate_metric_names():
    with pytest.raises(ValueError, match="duplicate"):
        EvaluationRecord(
            run_id="r",
            checkpoint_id="c",
            task_id="t",
            protocol_id="p",
            seed=1,
            episode_count=5,
            metrics=(
                MetricValue("success", 0.5, "fraction", "mean"),
                MetricValue("success", 0.3, "fraction", "mean"),
            ),
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_metrics_must_be_finite(value):
    with pytest.raises(ValueError):
        MetricValue("loss", value, "nats", "mean")


def test_positive_episode_count_required():
    with pytest.raises(ValueError, match="episode_count"):
        EvaluationRecord(
            run_id="r",
            checkpoint_id="c",
            task_id="t",
            protocol_id="p",
            seed=0,
            episode_count=0,
            metrics=(),
        )
