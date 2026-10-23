"""Generic online collection from the Environment contract."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from aeris.contracts.environment import Environment
from aeris.domain.experience import Transition
from aeris.domain.validation import require_int


@dataclass(frozen=True, slots=True)
class CollectionResult:
    transitions: tuple[Transition, ...]
    completed_episodes: int
    seed: int


def collect_transitions(
    env: Environment,
    *,
    count: int,
    seed: int,
    choose_action: Callable[[Any, np.random.Generator], Any],
) -> CollectionResult:
    """Collect exactly `count` transitions. No agent/framework internals required.

    Reset with `seed` once; later resets advance the environment's own RNG.
    The action selection RNG is separate and deterministic.
    """
    require_int("count", count, minimum=1)
    require_int("seed", seed)
    rng = np.random.default_rng(seed + 1)
    obs = env.reset(seed=seed).observation
    collected: list[Transition] = []
    completed = 0
    for _ in range(count):
        action = choose_action(obs, rng)
        result = env.step(action)
        collected.append(
            Transition(
                observation=obs,
                action=action,
                next_observation=result.observation,
                reward=result.reward,
                terminated=result.terminated,
                truncated=result.truncated,
            )
        )
        obs = result.observation
        if result.terminated or result.truncated:
            completed += 1
            obs = env.reset().observation
    return CollectionResult(transitions=tuple(collected), completed_episodes=completed, seed=seed)
