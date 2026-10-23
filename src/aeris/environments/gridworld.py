"""Deterministic four-room grid with one-hot observations.

This is an AERIS diagnostic environment, not a claim to reproduce the
maze layouts, data collection or reward conventions in the FB paper.
Coordinates are (row, column). Actions: 0=up, 1=down, 2=left, 3=right.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Iterator
from typing import Any
import numpy as np

from aeris.domain.environment import ResetResult, StepResult
from aeris.domain.experience import Transition


class FourRooms:
    """Small tabular environment with AERIS environment contract.

    The observation is always a 121-dimensional one-hot vector. Reward is
    intentionally None: goal rewards are defined only by evaluation tasks.
    """
    WIDTH = 11
    HEIGHT = 11
    ACTIONS: tuple[tuple[int, int], ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))

    def __init__(self, *, max_episode_steps: int = 100, start_position: tuple[int, int] | None = None):
        if type(max_episode_steps) is not int or max_episode_steps < 1:
            raise ValueError("max_episode_steps must be a positive integer")
        self.max_episode_steps = max_episode_steps
        self.num_actions = len(self.ACTIONS)
        self.observation_dim = self.WIDTH * self.HEIGHT
        # Walls in center row and column, four doors between the rooms.
        self._walls = {(5, i) for i in range(self.WIDTH)} | {(i, 5) for i in range(self.HEIGHT)}
        self._walls -= {(5, 2), (5, 8), (2, 5), (8, 5)}
        self._walls |= {(0, i) for i in range(self.WIDTH)} | {(10, i) for i in range(self.WIDTH)}
        self._walls |= {(i, 0) for i in range(self.HEIGHT)} | {(i, 10) for i in range(self.HEIGHT)}
        self.positions = tuple((r, c) for r in range(self.HEIGHT) for c in range(self.WIDTH) if (r, c) not in self._walls)
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
        self._position = (self.start_position if self.start_position is not None else self.positions[int(self._rng.integers(len(self.positions)))])
        self._steps = 0
        self._done = False
        return ResetResult(observation=self.observation_for(self._position), info={"position": self._position})

    def step(self, action: Any) -> StepResult:
        if self._position is None or self._done:
            raise RuntimeError("call reset() before stepping a new episode")
        self._position = self.next_position(self._position, action)
        self._steps += 1
        self._done = self._steps >= self.max_episode_steps
        return StepResult(observation=self.observation_for(self._position), reward=None, terminated=False, truncated=self._done, info={"position": self._position, "steps": self._steps})

    def all_transitions(self) -> Iterator[Transition]:
        """Oracle full-coverage dataset. Never used implicitly by the runner."""
        for pos in self.positions:
            for action in range(self.num_actions):
                yield Transition(observation=self.observation_for(pos), action=action, next_observation=self.observation_for(self.next_position(pos, action)))

    def shortest_distance(self, start: tuple[int, int], goal: tuple[int, int]) -> int:
        """Exact tabular oracle distance for later diagnostic comparisons."""
        self._check_position(start)
        self._check_position(goal)
        if start == goal:
            return 0
        queue: deque[tuple[tuple[int, int], int]] = deque([(start, 0)])
        visited = {start}
        while queue:
            position, dist = queue.popleft()
            for action in range(self.num_actions):
                nxt = self.next_position(position, action)
                if nxt == goal:
                    return dist + 1
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append((nxt, dist + 1))
        raise RuntimeError("goal is unreachable: invalid four-room layout")

    def close(self) -> None:
        pass
