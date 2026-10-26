"""Exact models of small deterministic environments with enumerable states.

Arrays are validated once, here, so the analyses that consume these models
(value iteration, exact occupancies, policy comparison) can assume valid input.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from math import isfinite

import numpy as np

from aeris.domain.artifacts import DiscreteState


def _frozen(array: np.ndarray) -> np.ndarray:
    array = np.array(array)  # private copy
    array.flags.writeable = False
    return array


@dataclass(frozen=True, eq=False)
class TabularDynamics:
    """Reward-free deterministic dynamics: ``next_states[s, a]`` is the successor index.

    Implements the ``TransitionTable`` contract.
    """

    states: tuple[DiscreteState, ...]
    next_states: np.ndarray
    _index: dict[DiscreteState, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        states = tuple(tuple(int(x) for x in s) for s in self.states)
        table = np.asarray(self.next_states)
        if table.ndim != 2 or table.shape[0] != len(states) or table.shape[1] < 1:
            raise ValueError("next_states must have shape [states, actions] matching the states")
        if table.dtype.kind not in "iu" or np.any((table < 0) | (table >= len(states))):
            raise ValueError("next_states must contain valid integer state indices")
        index = {state: i for i, state in enumerate(states)}
        if len(index) != len(states):
            raise ValueError("states must be unique")
        object.__setattr__(self, "states", states)
        object.__setattr__(self, "next_states", _frozen(table.astype(np.int64)))
        object.__setattr__(self, "_index", index)

    @property
    def num_states(self) -> int:
        return len(self.states)

    @property
    def num_actions(self) -> int:
        return self.next_states.shape[1]

    def index_of(self, state: DiscreteState) -> int:
        try:
            return self._index[tuple(state)]
        except KeyError:
            raise ValueError(f"unknown state: {state!r}") from None

    def shortest_path_lengths(self, target: int) -> np.ndarray:
        """Fewest steps from every state to ``target``; -1 where it is unreachable."""
        predecessors: list[set[int]] = [set() for _ in self.states]
        for s, successors in enumerate(self.next_states):
            for nxt in successors:
                predecessors[int(nxt)].add(s)
        lengths = np.full(self.num_states, -1, dtype=np.int64)
        lengths[target] = 0
        queue = deque([target])
        while queue:
            current = queue.popleft()
            for previous in predecessors[current]:
                if lengths[previous] < 0:
                    lengths[previous] = lengths[current] + 1
                    queue.append(previous)
        return lengths

    def require_full_coverage(
        self, samples: Iterable[tuple[DiscreteState, int, DiscreteState]]
    ) -> None:
        """Require exactly one correct ``(state, action, next_state)`` sample per state-action."""
        needed = self.num_states * self.num_actions
        seen: set[tuple[int, int]] = set()
        count = 0
        for state, action, next_state in samples:
            count += 1
            s = self.index_of(state)
            if not 0 <= action < self.num_actions:
                raise ValueError("dataset contains invalid action")
            if self.index_of(next_state) != self.next_states[s, action]:
                raise ValueError("dataset contains incorrect transition")
            seen.add((s, action))
        if count != needed or len(seen) != needed:
            raise ValueError("dataset contains duplicate or missing state-actions")


@dataclass(frozen=True, eq=False)
class TabularMDP:
    """Deterministic dynamics with rewards on transition, absorbing terminals and a discount."""

    dynamics: TabularDynamics
    rewards: np.ndarray
    terminals: np.ndarray
    gamma: float

    def __post_init__(self) -> None:
        shape = self.dynamics.next_states.shape
        rewards = np.asarray(self.rewards, dtype=np.float64)
        terminals = np.asarray(self.terminals)
        if rewards.shape != shape or not np.isfinite(rewards).all():
            raise ValueError("rewards must be finite with shape [states, actions]")
        if terminals.shape != (shape[0],) or terminals.dtype.kind != "b":
            raise ValueError("terminals must be a boolean vector with one entry per state")
        if not isfinite(self.gamma) or not 0 <= self.gamma < 1:
            raise ValueError("gamma must lie in [0, 1)")
        object.__setattr__(self, "rewards", _frozen(rewards))
        object.__setattr__(self, "terminals", _frozen(terminals))

    @classmethod
    def goal_reaching(cls, dynamics: TabularDynamics, goal: int, *, gamma: float) -> TabularMDP:
        """Reward 1 on ENTERING ``goal``, which is then terminal; 0 otherwise.

        Waiting at the goal earns nothing, and the reward is a diagnostic task
        definition, not something assumed present in reward-free training data.
        """
        rewards = (dynamics.next_states == goal).astype(np.float64)
        rewards[goal] = 0.0
        terminals = np.zeros(dynamics.num_states, dtype=bool)
        terminals[goal] = True
        return cls(dynamics, rewards, terminals, gamma)
