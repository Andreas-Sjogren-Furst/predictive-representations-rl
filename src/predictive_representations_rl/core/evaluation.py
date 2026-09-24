from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from .backend import Policy
from .data import Observation


class Environment(Protocol):
    def reset(self, *, seed: int | None = None) -> tuple[Observation, dict[str, Any]]:
        ...

    def step(self, action: np.ndarray) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        ...

@dataclass(frozen=True)
class EvaluationResult:
    episode_returns: np.ndarray
    episode_lengths: np.ndarray

    @property
    def mean_return(self) -> float:
        return float(np.mean(self.episode_returns))

    @property
    def std_return(self) -> float:
        return float(np.std(self.episode_returns))

    @property
    def mean_length(self) -> float:
        return float(np.mean(self.episode_lengths))

def evaluate(policy: Policy, env: Environment, *, num_episodes: int, seed: int = 0) -> EvaluationResult:
    returns = []
    lengths = []

    for episode in range(num_episodes):
        observation, _ = env.reset(seed=seed + episode)
        policy_state = policy.initial_state()

        episode_return = 0.0
        episode_length = 0

        while True:
            action, policy_state = policy.act(observation, policy_state, deterministic=True)
            observation,reward,terminated,truncated = env.step(action)

            episode_return += float(reward)
            episode_length += 1

            if terminated or truncated:
                break

        returns.append(episode_return)
        lengths.append(episode_length)

    return EvaluationResult(
        episode_returns=np.asarray(returns, dtype=np.float64),
        episode_lengths=np.asarray(lengths, dtype=np.int64)
    )
