import json
from dataclasses import FrozenInstanceError

import pytest

from aeris.domain import (
    ComponentSpec,
    EvaluationPlan,
    EvaluationTarget,
    ExperimentSpec,
    TaskEvidenceRef,
    TaskSpec,
    TrainingPlan,
)


def make_spec(**changes):
    fields = dict(
        name="demo",
        agent=ComponentSpec("fb", {"latent_dim": 32}),
        dataset=ComponentSpec("sampled_dataset"),
        training=TrainingPlan("offline", ComponentSpec("offline_updates", {"updates": 100})),
        seed=17,
        evaluation=EvaluationPlan(
            ComponentSpec("zero_shot_episodic"),
            (EvaluationTarget(TaskSpec("goal_a"), TaskEvidenceRef("reward_samples", "inference")),),
        ),
    )
    fields.update(changes)
    return ExperimentSpec(**fields)


def test_offline_spec_can_be_serialized_without_ml_dependencies():
    spec = make_spec()
    exported = spec.to_dict()
    assert json.loads(json.dumps(exported))["training"]["strategy"]["settings"]["updates"] == 100
    assert exported["evaluation"]["targets"][0]["inference_evidence"]["split"] == "inference"


def test_component_config_is_deeply_frozen_and_snapshotted():
    original = {"a": [1, {"b": 2}]}
    component = ComponentSpec("fb", original)
    original["a"][1]["b"] = 999
    assert component.settings["a"][1]["b"] == 2
    with pytest.raises(TypeError):
        component.settings["a"][1]["b"] = 9
    with pytest.raises(TypeError):
        component.settings["a"] = 4
    with pytest.raises(FrozenInstanceError):
        component.name = "other"


def test_export_is_independent_copy():
    spec = make_spec()
    exported = spec.to_dict()
    exported["agent"]["settings"]["latent_dim"] = 999
    assert spec.agent.settings["latent_dim"] == 32


@pytest.mark.parametrize("invalid", [{"x": object()}, {"x": float("nan")}, {7: "no"}])
def test_config_rejects_non_json_values(invalid):
    with pytest.raises((TypeError, ValueError)):
        ComponentSpec("bad", invalid)


def test_training_modes_require_appropriate_sources():
    offline = make_spec()
    assert offline.environment is None
    with pytest.raises(ValueError, match="dataset"):
        make_spec(dataset=None)
    with pytest.raises(ValueError, match="environment"):
        make_spec(training=TrainingPlan("online", ComponentSpec("online_collection")))
    with pytest.raises(ValueError, match="checkpoint"):
        make_spec(training=TrainingPlan("none"))
    eval_only = make_spec(training=TrainingPlan("none"), initial_checkpoint="checkpoint_7")
    assert eval_only.initial_checkpoint == "checkpoint_7"


def test_training_strategy_is_explicit():
    with pytest.raises(ValueError):
        TrainingPlan("offline")
    with pytest.raises(ValueError):
        TrainingPlan("none", ComponentSpec("should_not_exist"))


def test_evaluation_targets_must_not_duplicate_task_identity():
    with pytest.raises(ValueError, match="duplicate"):
        EvaluationPlan(
            ComponentSpec("eval"),
            (EvaluationTarget(TaskSpec("goal_a")), EvaluationTarget(TaskSpec("goal_a"))),
        )


def test_seed_cannot_be_boolean():
    with pytest.raises(ValueError):
        make_spec(seed=True)


def test_switch_environment_changes_spec_not_structure():
    grid = make_spec(environment=ComponentSpec("four_rooms"))
    other = make_spec(environment=ComponentSpec("continuous_maze"))
    assert grid.agent == other.agent
    assert grid.training == other.training
    assert grid.environment.name != other.environment.name
    # This is declarative substitution, not yet an executable compatibility test.


def test_same_task_can_use_different_inference_evidence():
    plan = EvaluationPlan(
        ComponentSpec("eval"),
        (
            EvaluationTarget(TaskSpec("goal_a"), TaskEvidenceRef("dataset_a", "inference")),
            EvaluationTarget(TaskSpec("goal_a"), TaskEvidenceRef("dataset_b", "inference")),
        ),
    )
    assert len(plan.targets) == 2
