"""Run with: PYTHONPATH=src python examples/specification.py.

Demonstrates domain construction only; this step does not execute training.
"""
import json

from aeris.domain import (
    ComponentSpec,
    EvaluationPlan,
    EvaluationTarget,
    ExperimentSpec,
    TaskEvidenceRef,
    TaskSpec,
    TrainingPlan,
)

spec = ExperimentSpec(
    name="fb_four_rooms_offline",
    agent=ComponentSpec("fb", {"latent_dim": 32}),
    dataset=ComponentSpec("four_rooms_offline_v1", {"split": "train"}),
    environment=ComponentSpec("four_rooms", {"observation": "one_hot"}),
    training=TrainingPlan(
        mode="offline",
        strategy=ComponentSpec("offline_updates", {"updates": 1000}),
    ),
    evaluation=EvaluationPlan(
        protocol=ComponentSpec("episodic_zero_shot", {"episodes_per_task": 10}),
        targets=(
            EvaluationTarget(
                task=TaskSpec("reach_cell_42", {"target_cell": 42}),
                inference_evidence=TaskEvidenceRef(
                    source_id="four_rooms_reward_examples_v1", split="inference"
                ),
            ),
        ),
    ),
    seed=42,
)
print(json.dumps(spec.to_dict(), indent=2))
