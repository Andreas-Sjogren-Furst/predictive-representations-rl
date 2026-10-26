"""Record a policy's deterministic choice in every enumerated state."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from aeris.contracts.agent import Agent
from aeris.domain.agent import ActionRequest
from aeris.domain.artifacts import DiscreteState, PolicySnapshot


def record_policy_snapshot(
    agent: Agent,
    states: Sequence[DiscreteState],
    observe: Callable[[DiscreteState], Any],
    *,
    condition: Any,
) -> PolicySnapshot:
    """Query ``agent.act`` once per state; uses only the public Agent contract."""
    actions = [
        agent.act(
            ActionRequest(observation=observe(state), condition=condition, deterministic=True)
        ).action
        for state in states
    ]
    return PolicySnapshot(states=tuple(states), actions=tuple(actions))
