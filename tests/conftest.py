"""Shared fixtures that write real, schema-valid run artifacts."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from aeris.agents.fb.representations import FBRepresentations
from aeris.data import RunDirectory
from aeris.domain import (
    ComponentSpec,
    EvaluationPlan,
    EvaluationTarget,
    ExperimentSpec,
    RunManifest,
    TaskSpec,
    TrainingPlan,
)


def write_four_rooms_run(
    root: Path,
    *,
    run_id: str = "run-42",
    status: str = "completed",
    environment: str = "four_rooms",
    goal: tuple[int, int] = (9, 9),
    max_steps: int = 10,
    dataset_settings: Mapping[str, Any] | None = None,
) -> RunDirectory:
    """Write the manifest of a single-goal Four Rooms run, as the training example does."""
    spec = ExperimentSpec(
        name="fb_demo",
        agent=ComponentSpec("discrete_fb"),
        environment=ComponentSpec(environment),
        dataset=ComponentSpec("fixed_grid_transitions", dataset_settings or {"origin": "oracle"}),
        training=TrainingPlan("offline", ComponentSpec("fixed_batch_updates")),
        evaluation=EvaluationPlan(
            protocol=ComponentSpec("goal_reaching_v1", {"max_steps": max_steps}),
            targets=(
                EvaluationTarget(
                    TaskSpec(
                        f"reach_{goal[0]}_{goal[1]}", {"goal_row": goal[0], "goal_col": goal[1]}
                    )
                ),
            ),
        ),
        seed=42,
    )
    run = RunDirectory(root)
    run.write_manifest(
        RunManifest(
            run_id=run_id, experiment=spec, status=status, selected_checkpoint="checkpoint.pt"
        )
    )
    return run


@pytest.fixture
def four_rooms_run(tmp_path) -> Callable[..., RunDirectory]:
    """Factory: ``four_rooms_run(**overrides)`` writes a run under ``tmp_path / "run"``."""

    def make(subdir: str = "run", **overrides: Any) -> RunDirectory:
        return write_four_rooms_run(tmp_path / subdir, **overrides)

    return make


def synthetic_representations(
    states=((1, 1), (1, 2), (2, 1)), goals=((1, 2),), d=4, a=2, seed=0
) -> FBRepresentations:
    """Well-formed random FB representations (no torch needed)."""
    rng = np.random.default_rng(seed)
    s, g = len(states), len(goals)
    policy = rng.random((g, s, a))
    return FBRepresentations(
        run_id="run",
        checkpoint="checkpoint.pt",
        gamma=0.99,
        states=states,
        goals=goals,
        backward=rng.normal(size=(s, d)).astype(np.float32),
        goal_conditions=rng.normal(size=(g, d)).astype(np.float32),
        forward=rng.normal(size=(g, s, d, a)).astype(np.float32),
        target_policy=(policy / policy.sum(-1, keepdims=True)).astype(np.float32),
        state_density=np.full(s, 1 / s, dtype=np.float32),
        exact_density=rng.random((g, s, a, s)).astype(np.float32),
    )


def synthetic_suite_report(task_count_per_family: int = 2, seed: int = 0):
    """A well-formed ZeroShotSuiteReport over a few tasks of each four_rooms_v1 family."""
    from aeris.domain import SuiteOutcome, SuiteTaskResult, ZeroShotSuiteReport, task_family
    from aeris.environments import FourRooms
    from aeris.environments.four_rooms_tasks import four_rooms_v1

    rng = np.random.default_rng(seed)
    states = FourRooms().positions
    suite = four_rooms_v1()
    picked, seen = [], {}
    for task in suite.tasks:
        family = task_family(task)
        if seen.get(family, 0) < task_count_per_family:
            seen[family] = seen.get(family, 0) + 1
            picked.append(task)
    protocols = {"dataset_rewards": "relabelled data", "all_states": "every state"}
    results = []
    for task in picked:
        family = task_family(task)
        outcomes = {
            protocol: SuiteOutcome(
                normalized_score=float(rng.uniform(0, 1)),
                policy_value=1.0,
                goal_success=float(rng.uniform(0, 1)) if family == "goal" else None,
                evidence_samples=272,
                rewarded_samples=0 if (protocol == "dataset_rewards" and task is picked[0]) else 4,
                actions=tuple(int(a) for a in rng.integers(4, size=len(states))),
            )
            for protocol in protocols
        }
        results.append(
            SuiteTaskResult(
                task=task, family=family, optimal_value=2.0, random_value=0.5, outcomes=outcomes
            )
        )
    means = {"all": {p: 0.5 for p in protocols}, **{f: {p: 0.5 for p in protocols} for f in seen}}
    return ZeroShotSuiteReport(
        run_id="run-42",
        checkpoint="checkpoint.pt",
        suite=suite.name,
        suite_description=suite.description,
        gamma=0.99,
        max_steps=60,
        evidence_source="dataset.npz sha256:abc",
        evidence_protocols=protocols,
        states=states,
        tasks=tuple(results),
        mean_score=means,
    )
