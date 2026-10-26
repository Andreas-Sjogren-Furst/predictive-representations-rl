"""Deterministic four-room grid with one-hot observations.

This is an AERIS diagnostic environment, not a claim to reproduce the
maze layouts, data collection or reward conventions in the FB paper.
Coordinates are (row, column). Actions: 0=up, 1=down, 2=left, 3=right.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from functools import cached_property
from typing import Any

import numpy as np

from aeris.domain.environment import ResetResult, StepResult
from aeris.domain.experience import Transition
from aeris.domain.spaces import DiscreteActionSpec, ObservationSpec, SpaceSpec
from aeris.domain.task import TaskSpec
from aeris.domain.validation import require_int
from aeris.environments.tabular import TabularDynamics

Position = tuple[int, int]


class FourRooms:
    """Small tabular environment implementing the Environment and TabularModel contracts.

    The observation is always a 121-dimensional one-hot vector. Reward is
    intentionally None: goal rewards are defined only by evaluation tasks.
    States of the exact model are the traversable ``positions``.
    """

    WIDTH = 11
    HEIGHT = 11
    ACTIONS: tuple[tuple[int, int], ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))

    def __init__(
        self, *, max_episode_steps: int = 100, start_position: tuple[int, int] | None = None
    ):
        require_int("max_episode_steps", max_episode_steps, minimum=1)
        self.max_episode_steps = max_episode_steps
        self.num_actions = len(self.ACTIONS)
        self.observation_dim = self.WIDTH * self.HEIGHT
        # Walls in center row and column, four doors between the rooms.
        self._walls = {(5, i) for i in range(self.WIDTH)} | {(i, 5) for i in range(self.HEIGHT)}
        self._walls -= {(5, 2), (5, 8), (2, 5), (8, 5)}
        self._walls |= {(0, i) for i in range(self.WIDTH)} | {(10, i) for i in range(self.WIDTH)}
        self._walls |= {(i, 0) for i in range(self.HEIGHT)} | {(i, 10) for i in range(self.HEIGHT)}
        self.positions = tuple(
            (r, c)
            for r in range(self.HEIGHT)
            for c in range(self.WIDTH)
            if (r, c) not in self._walls
        )
        if start_position is not None:
            self._check_position(start_position)
        self.start_position = start_position
        self._rng = np.random.default_rng()
        self._position: tuple[int, int] | None = None
        self._steps = 0
        self._done = False

    def _check_position(self, pos: tuple[int, int]) -> None:
        if pos not in self.positions:
            raise ValueError(f"not a traversable grid position: {pos!r}")

    def observation_for(self, pos: tuple[int, int]) -> np.ndarray:
        self._check_position(pos)
        obs = np.zeros(self.observation_dim, dtype=np.float32)
        obs[pos[0] * self.WIDTH + pos[1]] = 1.0
        return obs

    def position_of(self, observation: Any) -> tuple[int, int]:
        arr = np.asarray(observation)
        if arr.shape != (self.observation_dim,) or not np.isfinite(arr).all():
            raise ValueError("expected a finite one-hot grid observation")
        ix = int(arr.argmax())
        if arr[ix] != 1 or np.count_nonzero(arr) != 1:
            raise ValueError("expected a one-hot grid observation")
        pos = divmod(ix, self.WIDTH)
        self._check_position(pos)
        return pos

    def next_position(self, pos: tuple[int, int], action: int) -> tuple[int, int]:
        self._check_position(pos)
        if type(action) is not int or not 0 <= action < self.num_actions:
            raise ValueError(f"action must be an integer in [0, {self.num_actions})")
        dr, dc = self.ACTIONS[action]
        candidate = (pos[0] + dr, pos[1] + dc)
        return pos if candidate in self._walls else candidate

    def reset(self, *, seed: int | None = None) -> ResetResult:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._position = (
            self.start_position
            if self.start_position is not None
            else self.positions[int(self._rng.integers(len(self.positions)))]
        )
        self._steps = 0
        self._done = False
        return ResetResult(
            observation=self.observation_for(self._position), info={"position": self._position}
        )

    def step(self, action: Any) -> StepResult:
        if self._position is None or self._done:
            raise RuntimeError("call reset() before stepping a new episode")
        self._position = self.next_position(self._position, action)
        self._steps += 1
        self._done = self._steps >= self.max_episode_steps
        return StepResult(
            observation=self.observation_for(self._position),
            reward=None,
            terminated=False,
            truncated=self._done,
            info={"position": self._position, "steps": self._steps},
        )

    def all_transitions(self) -> Iterator[Transition]:
        """Oracle full-coverage dataset. Never used implicitly by the runner."""
        for pos in self.positions:
            for action in range(self.num_actions):
                yield Transition(
                    observation=self.observation_for(pos),
                    action=action,
                    next_observation=self.observation_for(self.next_position(pos, action)),
                )

    def shortest_distance(self, start: Position, goal: Position) -> int:
        """Exact tabular oracle distance for diagnostic comparisons."""
        dynamics = self.tabular_dynamics()
        return int(
            dynamics.shortest_path_lengths(dynamics.index_of(goal))[dynamics.index_of(start)]
        )

    # Exact model and task conventions (TabularModel contract)
    @property
    def spec(self) -> SpaceSpec:
        return SpaceSpec(
            ObservationSpec(self.observation_dim), DiscreteActionSpec(self.num_actions)
        )

    def tabular_dynamics(self) -> TabularDynamics:
        return self._dynamics

    @cached_property
    def _dynamics(self) -> TabularDynamics:
        index = {pos: i for i, pos in enumerate(self.positions)}
        table = [
            [index[self.next_position(pos, a)] for a in range(self.num_actions)]
            for pos in self.positions
        ]
        return TabularDynamics(self.positions, np.asarray(table, dtype=np.int64))

    def goal_task(self, goal: Position) -> TaskSpec:
        """The evaluation task "reach ``goal``", in the settings format this environment reads."""
        self._check_position(goal)
        return TaskSpec(
            task_id=f"reach_{goal[0]}_{goal[1]}",
            settings={"goal_row": goal[0], "goal_col": goal[1]},
        )

    def goal_from_task(self, task: TaskSpec) -> Position:
        try:
            goal = (int(task.settings["goal_row"]), int(task.settings["goal_col"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"task {task.task_id!r} has no Four Rooms goal_row/goal_col") from exc
        self._check_position(goal)
        return goal

    def require_full_coverage(self, transitions: Iterable[Transition]) -> None:
        """Require exactly one correct one-step sample for every state-action pair."""
        self.tabular_dynamics().require_full_coverage(
            (self.position_of(t.observation), t.action, self.position_of(t.next_observation))
            for t in transitions
        )

    def close(self) -> None:
        pass
