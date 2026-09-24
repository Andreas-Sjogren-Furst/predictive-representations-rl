from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np


Observation: TypeAlias = np.ndarray | Mapping[str, np.ndarray]

def observation_length(observation: Observation) -> int:
    if isinstance(observation, np.ndarray):
        return len(observation)

    lengths = { len(value) for value in observation.values() }
    if len(lengths) != 1:
        raise ValueError(f"Observation fields have inconsistent lenghts {lengths}")

    return lengths.pop()

@dataclass(frozen=True)
class Trajectory:
    observations: Observation
    actions: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray

    def __post_init__(self) -> None:
        num_steps = len(self.actions)

        if observation_length(self.observations) != num_steps + 1:
            raise ValueError("A trajectory with T actions must contain T + 1 observations")

        if len(self.terminated) != num_steps:
            raise ValueError("terminated must have length T.")

        if len(self.truncated) != num_steps:
            raise ValueError("truncated must have length T.")

    @property
    def num_steps(self) -> int:
        return len(self.actions)

    @property
    def episode_ends(self) -> np.ndarray:
        return np.logical_or(self.terminated, self.truncated)

@dataclass(frozen=True)
class RepresentationDataset:
    trajectories: tuple[Trajectory, ...]

    @property
    def num_transitions(self) -> int:
        return sum(traj.num_steps for traj in self.trajectories)

@dataclass(frozen=True)
class RewardedTrajectory:
    trajectory: Trajectory
    rewards: np.ndarray

    def __post__init__(self) -> None:
        if len(self.rewards) != self.trajectory.num_steps:
            raise ValueError("Rewards must contain one value per transition")

@dataclass(frozen=True)
class TaskDataset:
    task_name: str
    trajectories: tuple[RewardedTrajectory, ...]

    @ property
    def num_transitions(self) -> int:
        return sum(traj.trajectory.num_steps for traj in self.trajectories)
