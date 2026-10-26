import json

import pytest

from aeris.domain import (
    ComponentSpec,
    EvaluationPlan,
    EvaluationRecord,
    EvaluationTarget,
    EvaluationTrajectory,
    ExperimentSpec,
    MetricValue,
    RunManifest,
    TaskEvidenceRef,
    TaskSpec,
    TrainingMetricRecord,
    TrainingPlan,
    TrajectoryStep,
)


def manifest() -> RunManifest:
    spec = ExperimentSpec(
        name="demo",
        agent=ComponentSpec("discrete_fb", {"representation_dim": 32}),
        environment=ComponentSpec("four_rooms", {"max_episode_steps": 60}),
        dataset=ComponentSpec("grid", {"origin": "oracle", "sha256": "abc"}),
        training=TrainingPlan("offline", ComponentSpec("updates", {"updates": 10})),
        evaluation=EvaluationPlan(
            protocol=ComponentSpec("goal_reaching_v1", {"max_steps": 60}),
            targets=(
                EvaluationTarget(
                    TaskSpec("reach_9_9", {"goal_row": 9, "goal_col": 9}),
                    TaskEvidenceRef("rewards", "inference"),
                ),
            ),
        ),
        seed=7,
    )
    return RunManifest("run-7", spec, status="completed", provenance={"tool": {"v": [1, 2]}})


def roundtrip(record):
    data = json.loads(json.dumps(record.to_dict()))
    return type(record).from_dict(data)


def test_nested_manifest_roundtrips_through_json():
    original = manifest()
    restored = roundtrip(original)
    assert restored == original
    assert restored.experiment.evaluation.targets[0].task.settings["goal_row"] == 9
    assert restored.experiment.evaluation.targets[0].inference_evidence.split == "inference"


def test_results_and_trajectories_roundtrip():
    metric = MetricValue("success_rate", 0.5, "fraction", "mean")
    assert roundtrip(EvaluationRecord("r", "c", "t", "p", 1, 2, (metric,))).metrics == (metric,)
    assert roundtrip(TrainingMetricRecord("r", "gradient_updates", 3, (metric,))).step == 3
    trajectory = EvaluationTrajectory(
        run_id="r",
        task_id="t",
        checkpoint_id="c",
        protocol_id="p",
        seed=1,
        episode_index=0,
        initial_observation=[1.0, 0.0],
        steps=(TrajectoryStep([1.0, 0.0], 3, [0.0, 1.0], None, False, False, True),),
        success=True,
    )
    assert roundtrip(trajectory) == trajectory


def test_integer_json_numbers_are_accepted_for_floats():
    data = {
        "run_id": "r",
        "step_axis": "s",
        "step": 0,
        "metrics": [{"name": "m", "value": 1, "unit": "u", "aggregation": "a"}],
    }
    assert TrainingMetricRecord.from_dict(data).metrics[0].value == 1.0


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda d: d.update(unexpected=1), "unknown field"),
        (lambda d: d.pop("run_id"), "missing field"),
        (lambda d: d.update(status="paused"), "expected one of"),
        (lambda d: d["experiment"].update(seed="7"), "RunManifest.experiment.seed"),
        (lambda d: d["experiment"]["agent"].update(name=""), "nonempty"),
    ],
)
def test_malformed_artifacts_are_rejected_with_a_path(mutate, message):
    data = manifest().to_dict()
    mutate(data)
    with pytest.raises(ValueError, match=message):
        RunManifest.from_dict(data)
