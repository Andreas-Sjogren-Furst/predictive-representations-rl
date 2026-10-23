"""Runtime episode traces for observational analysis.

Unlike ExperimentSpec, these records deliberately accept backend-native
observations and actions; serialization belongs to the artifact adapter.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class TrajectoryStep:
    observation: Any
    action: Any
    next_observation: Any
    reward: float | None
    terminated: bool
    truncated: bool
    goal_reached: bool


@dataclass(frozen=True, slots=True)
class EvaluationTrajectory:
    run_id: str
    task_id: str
    checkpoint_id: str
    protocol_id: str
    seed: int
    episode_index: int
    initial_observation: Any
    steps: tuple[TrajectoryStep, ...]
    success: bool

    @property
    def length(self) -> int:
        return len(self.steps)
