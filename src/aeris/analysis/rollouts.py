"""Exact rollouts of deterministic policies on known deterministic dynamics.

A deterministic policy on deterministic dynamics gives every state exactly one
successor, so its future from any start is a path that either enters the goal
or falls into a cycle it never leaves. Both are found exactly, with no sampling
and no step budget: a rollout stops at the goal or at the first repeated state.

Works on any validated ``TabularDynamics``; nothing here knows about a
specific environment, a run directory or a file format.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aeris.environments.tabular import TabularDynamics

# Goal outcomes of a start state (``PolicyFates.goal_outcomes``).
AT_GOAL, OPTIMAL, DETOUR, LATE, TRAPPED = "at goal", "optimal", "detour", "late", "trapped"
GOAL_OUTCOMES = (AT_GOAL, OPTIMAL, DETOUR, LATE, TRAPPED)


def successors(dynamics: TabularDynamics, actions: np.ndarray) -> np.ndarray:
    """The one successor of every state under one deterministic action per state."""
    actions = np.asarray(actions)
    if actions.shape != (dynamics.num_states,) or actions.dtype.kind not in "iu":
        raise ValueError("actions must hold one integer action per state")
    if np.any((actions < 0) | (actions >= dynamics.num_actions)):
        raise ValueError("actions must be valid action indices")
    return dynamics.next_states[np.arange(dynamics.num_states), actions]


@dataclass(frozen=True, slots=True)
class Rollout:
    """The exact future of a deterministic policy from ``states[0]``.

    ``states`` runs until the goal is entered or a state repeats. In the second
    case the last entry repeats ``states[cycle_start]``, and from there on the
    policy loops through ``cycle`` forever.
    """

    states: tuple[int, ...]
    actions: tuple[int, ...]  # actions[i] is taken in states[i]; one fewer than states
    goal_step: int | None  # steps to enter the goal (0 when starting there); None if never
    cycle_start: int | None  # index into states where the cycle begins; None if the goal ends it

    @property
    def start(self) -> int:
        return self.states[0]

    @property
    def steps(self) -> int:
        return len(self.actions)

    @property
    def cycle(self) -> tuple[int, ...]:
        """The states the policy loops through forever; empty when the goal ends the rollout."""
        return () if self.cycle_start is None else self.states[self.cycle_start : -1]

    @property
    def reached_goal(self) -> bool:
        return self.goal_step is not None


def rollout(
    dynamics: TabularDynamics, actions: np.ndarray, start: int, *, goal: int | None = None
) -> Rollout:
    """Follow one action per state from ``start`` until the goal or a repeated state."""
    nxt = successors(dynamics, actions)
    if not 0 <= start < dynamics.num_states:
        raise ValueError("start must be a valid state index")
    if goal is not None and not 0 <= goal < dynamics.num_states:
        raise ValueError("goal must be a valid state index")
    states, position = [start], {start: 0}
    if start == goal:
        return Rollout((start,), (), 0, None)
    while True:
        current = int(nxt[states[-1]])
        states.append(current)
        if current == goal:
            return Rollout(tuple(states), _actions(actions, states), len(states) - 1, None)
        if current in position:
            return Rollout(tuple(states), _actions(actions, states), None, position[current])
        position[current] = len(states) - 1


def _actions(actions: np.ndarray, states: list[int]) -> tuple[int, ...]:
    return tuple(int(actions[s]) for s in states[:-1])


@dataclass(frozen=True, slots=True)
class PolicyFates:
    """Where a deterministic policy leads from every start state.

    ``cycles`` lists every cycle some start falls into (each rotated to begin at
    its smallest state); ``cycle_of[s]`` indexes it, -1 where the goal comes
    first. ``visits[s]`` counts the other start states whose rollout enters ``s``.
    """

    goal: int | None
    goal_steps: np.ndarray  # [state] steps to enter the goal, -1 if never (0 at the goal)
    shortest_steps: np.ndarray | None  # [state] fewest possible steps to the goal, -1 if none
    cycles: tuple[tuple[int, ...], ...]
    cycle_of: np.ndarray  # [state]
    transient: np.ndarray  # [state] steps before entering the cycle, or to the goal
    visits: np.ndarray  # [state]

    def goal_outcomes(self, *, max_steps: int) -> np.ndarray:
        """Each start's outcome: at goal, optimal, detour, late or trapped.

        A detour reaches the goal in more than the fewest possible steps; it is
        late when that also exceeds ``max_steps``. A shortest path is optimal
        even when the goal lies beyond ``max_steps``, as nothing could be faster.
        """
        if self.goal is None or self.shortest_steps is None:
            raise ValueError("goal outcomes need a goal")
        steps, shortest = self.goal_steps, self.shortest_steps
        outcomes = np.full(len(steps), TRAPPED, dtype=object)
        outcomes[(steps >= 0) & (steps > max_steps)] = LATE
        outcomes[(steps >= 0) & (steps <= max_steps) & (steps > shortest)] = DETOUR
        outcomes[(steps >= 0) & (steps == shortest)] = OPTIMAL
        outcomes[self.goal] = AT_GOAL
        return outcomes


def policy_fates(
    dynamics: TabularDynamics, actions: np.ndarray, *, goal: int | None = None
) -> PolicyFates:
    """Roll the policy out exactly from every start state."""
    count = dynamics.num_states
    goal_steps = np.full(count, -1, dtype=np.int64)
    cycle_of = np.full(count, -1, dtype=np.int64)
    transient = np.zeros(count, dtype=np.int64)
    visits = np.zeros(count, dtype=np.int64)
    cycles: dict[tuple[int, ...], int] = {}
    for start in range(count):
        path = rollout(dynamics, actions, start, goal=goal)
        visits[list(set(path.states[1:]) - {start})] += 1
        if path.goal_step is not None:
            goal_steps[start] = transient[start] = path.goal_step
            continue
        cycle = path.cycle
        lowest = cycle.index(min(cycle))
        cycle_of[start] = cycles.setdefault(cycle[lowest:] + cycle[:lowest], len(cycles))
        transient[start] = path.cycle_start or 0
    return PolicyFates(
        goal=goal,
        goal_steps=goal_steps,
        shortest_steps=None if goal is None else dynamics.shortest_path_lengths(goal),
        cycles=tuple(cycles),
        cycle_of=cycle_of,
        transient=transient,
        visits=visits,
    )
