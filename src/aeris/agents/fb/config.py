"""Discrete-action forward-backward hyperparameters.

Defaults mirror Touati & Ollivier's published discrete-maze command where
possible; dimensions and action count come from the selected environment.
"""

from dataclasses import dataclass
from typing import Literal

TaskSampling = Literal["cauchy_ball", "uniform_ball", "goal_oriented"]


@dataclass(frozen=True, slots=True)
class DiscreteFBConfig:
    observation_dim: int
    num_actions: int
    representation_dim: int = 100
    hidden_dim: int = 256
    learning_rate: float = 5e-4
    gamma: float = 0.99
    polyak: float = 0.95  # retention coefficient for target networks
    regularization_coefficient: float = 1.0
    temperature: float = 200.0
    exploration_epsilon: float = 1.0
    task_sampling: TaskSampling = "cauchy_ball"
    seed: int = 0
    device: str = "cpu"

    def __post_init__(self) -> None:
        for key in ("observation_dim", "num_actions", "representation_dim", "hidden_dim"):
            val = getattr(self, key)
            if (
                isinstance(val, bool)
                or not isinstance(val, int)
                or val < (2 if key == "num_actions" else 1)
            ):
                raise ValueError(f"{key} must be a positive integer (num_actions >= 2)")
        if not (0 <= self.gamma < 1):
            raise ValueError("gamma must be in [0, 1)")
        if not (0 <= self.polyak < 1):
            raise ValueError("polyak must be in [0, 1)")
        if self.learning_rate <= 0 or self.temperature <= 0:
            raise ValueError("learning_rate and temperature must be positive")
        if not (0 <= self.exploration_epsilon <= 1):
            raise ValueError("exploration_epsilon must be in [0, 1]")
        if self.regularization_coefficient < 0:
            raise ValueError("regularization_coefficient must be nonnegative")
        if self.task_sampling not in ("cauchy_ball", "uniform_ball", "goal_oriented"):
            raise ValueError(f"unsupported task sampling: {self.task_sampling}")
