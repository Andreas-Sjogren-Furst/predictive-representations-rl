"""Shared fixtures that write real, schema-valid run artifacts."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pytest

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
