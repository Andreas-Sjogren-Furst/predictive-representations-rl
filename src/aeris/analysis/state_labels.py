"""What is true of each Four Rooms state, as targets for probes and colours for maps.

Three groups: the grid's geometry, quantities of one goal task (exact), and the
dynamics. Continuous labels may have several outputs ([S, m]); categorical
labels hold class indices with names, and may accept several classes per state
(any optimal action counts as correct).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aeris.analysis.task_scores import optimal_solution
from aeris.environments.four_rooms import ROOMS, FourRooms, Position, room_of
from aeris.environments.tabular import TabularDynamics

GEOMETRY, TASK, DYNAMICS = "geometry", "task", "dynamics"
ACTION_NAMES = ("↑", "↓", "←", "→")  # FourRooms.ACTIONS order


@dataclass(frozen=True, eq=False)
class StateLabel:
    name: str
    group: str
    values: np.ndarray  # continuous [S] or [S, m]; categorical [S] class indices
    classes: tuple[str, ...] = ()  # names of the classes; empty for continuous labels
    accepted: np.ndarray | None = None  # [S, classes] bool: every class that counts as correct

    @property
    def categorical(self) -> bool:
        return bool(self.classes)

    @property
    def scalar(self) -> bool:
        """One value per state, so it can colour a map."""
        return self.values.ndim == 1


def _occupancy(dynamics: TabularDynamics, policy: np.ndarray, gamma: float) -> np.ndarray:
    """Normalized discounted state occupancy [S, S] from each start, the start included."""
    s = dynamics.num_states
    transition = np.zeros((s, s))
    for action in range(dynamics.num_actions):
        np.add.at(transition, (np.arange(s), dynamics.next_states[:, action]), policy[:, action])
    return (1 - gamma) * np.linalg.solve(np.eye(s) - gamma * transition, np.eye(s))


def four_rooms_labels(
    env: FourRooms, dynamics: TabularDynamics, *, goal: Position, gamma: float
) -> tuple[StateLabel, ...]:
    """Every probe target for ``goal``, in the order: geometry, task, dynamics."""
    states = dynamics.states
    s, a = dynamics.num_states, dynamics.num_actions
    rooms = (*ROOMS, "doorway")
    blocked = (dynamics.next_states == np.arange(s)[:, None]).sum(axis=1)
    goal_index = dynamics.index_of(goal)
    rewards = np.zeros(s)
    rewards[goal_index] = 1.0
    optimal = optimal_solution(dynamics, rewards, gamma=gamma)
    first_optimal = optimal.optimal_actions.argmax(axis=1)
    greedy = np.zeros((s, a))
    greedy[np.arange(s), first_optimal] = 1.0
    uniform = np.full((s, a), 1.0 / a)
    positions = np.array(states, dtype=np.float64)  # [S, 2] row, column
    next_positions = positions[dynamics.next_states].reshape(s, -1)  # [S, A * 2]
    return (
        StateLabel("row", GEOMETRY, np.array([p[0] for p in states], dtype=np.float64)),
        StateLabel("column", GEOMETRY, np.array([p[1] for p in states], dtype=np.float64)),
        StateLabel("room", GEOMETRY, np.array([rooms.index(room_of(p)) for p in states]), rooms),
        StateLabel(
            "next to a wall",
            GEOMETRY,
            (blocked > 0).astype(int),
            ("open", "next to a wall"),
        ),
        StateLabel(
            "distance to goal",
            TASK,
            dynamics.shortest_path_lengths(goal_index).astype(np.float64),
        ),
        StateLabel("V* (goal)", TASK, optimal.values),
        StateLabel(
            "optimal action",
            TASK,
            first_optimal,
            ACTION_NAMES[:a],
            accepted=optimal.optimal_actions,
        ),
        StateLabel("next positions", DYNAMICS, next_positions),
        StateLabel("occupancy (random walk)", DYNAMICS, _occupancy(dynamics, uniform, gamma)),
        StateLabel("occupancy (π*, goal)", DYNAMICS, _occupancy(dynamics, greedy, gamma)),
    )
