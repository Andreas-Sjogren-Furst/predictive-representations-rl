from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from aeris.domain.experience import Transition


@dataclass(frozen=True, slots=True)
class FBTrainingBatch:
    observations: Any
    actions: Any
    next_observations: Any
    comparison_observations: Any
    goal_observations: Any | None = None


class FBTransitionSampler:
    """Independent pairs of transitions, matching the upstream loss sampler.

    Works with any transition list exposing vector observations and integer actions.
    Goal observations are optional; supply an explicit goal pool for goal-oriented
    task sampling. Never infer goals from *evaluation* data implicitly.
    """

    def __init__(
        self,
        transitions: Sequence[Transition],
        *,
        batch_size: int,
        seed: int = 0,
        goal_observations: Any | None = None,
    ):
        if len(transitions) == 0 or batch_size < 1:
            raise ValueError("need a nonempty transition source and positive batch_size")
        self.transitions = tuple(transitions)
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)
        self.goal_observations = (
            None if goal_observations is None else np.asarray(goal_observations, dtype=np.float32)
        )
        if self.goal_observations is not None and (
            self.goal_observations.ndim != 2 or not len(self.goal_observations)
        ):
            raise ValueError("goal_observations must be a nonempty matrix")

    def next_batch(self) -> FBTrainingBatch:
        first = self.rng.integers(len(self.transitions), size=self.batch_size)
        second = self.rng.integers(len(self.transitions), size=self.batch_size)
        primary = [self.transitions[i] for i in first]
        others = [self.transitions[i] for i in second]
        goals = None
        if self.goal_observations is not None:
            goal_indices = self.rng.integers(len(self.goal_observations), size=self.batch_size)
            goals = self.goal_observations[goal_indices]
        return FBTrainingBatch(
            observations=np.stack([np.asarray(t.observation, dtype=np.float32) for t in primary]),
            actions=np.asarray([t.action for t in primary], dtype=np.int64),
            next_observations=np.stack(
                [np.asarray(t.next_observation, dtype=np.float32) for t in primary]
            ),
            comparison_observations=np.stack(
                [np.asarray(t.observation, dtype=np.float32) for t in others]
            ),
            goal_observations=goals,
        )

    def state_dict(self) -> dict[str, Any]:
        return {"rng_state": self.rng.bit_generator.state}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.rng.bit_generator.state = state["rng_state"]
