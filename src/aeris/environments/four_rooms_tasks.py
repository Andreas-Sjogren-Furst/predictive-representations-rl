"""Four Rooms task builders and the built-in task suites.

Every task is a ``TaskSpec`` whose ``settings["type"]`` names its reward
family; ``FourRooms.state_rewards`` turns it into a reward per state.
"""

from __future__ import annotations

from aeris.domain.task import TaskSpec, TaskSuite
from aeris.environments.four_rooms import ROOMS, FourRooms, Position

# Two cells per room: its centre and its outer corner.
DISTANCE_GOALS: tuple[Position, ...] = (
    (2, 2),
    (2, 8),
    (8, 2),
    (8, 8),
    (1, 1),
    (1, 9),
    (9, 1),
    (9, 9),
)


def goal_task(goal: Position) -> TaskSpec:
    return FourRooms().goal_task(goal)


def room_task(room: str) -> TaskSpec:
    return TaskSpec(
        task_id=f"room_{room.replace('-', '_')}", settings={"type": "room", "room": room}
    )


def distance_task(goal: Position) -> TaskSpec:
    return TaskSpec(
        task_id=f"distance_{goal[0]}_{goal[1]}",
        settings={"type": "distance", "goal_row": goal[0], "goal_col": goal[1]},
    )


def random_sparse_task(seed: int, *, cells: int = 3) -> TaskSpec:
    return TaskSpec(
        task_id=f"random_sparse_{seed}",
        settings={"type": "random_sparse", "seed": seed, "cells": cells},
    )


def random_dense_task(seed: int) -> TaskSpec:
    return TaskSpec(task_id=f"random_dense_{seed}", settings={"type": "random_dense", "seed": seed})


def four_rooms_v1() -> TaskSuite:
    env = FourRooms()
    return TaskSuite(
        name="four_rooms_v1",
        description=(
            "Every goal cell (68); each room (4); negative distance to 8 goals; "
            "20 random 3-cell sparse rewards; 20 random dense Gaussian rewards"
        ),
        tasks=(
            *(goal_task(pos) for pos in env.positions),
            *(room_task(room) for room in ROOMS),
            *(distance_task(goal) for goal in DISTANCE_GOALS),
            *(random_sparse_task(seed) for seed in range(20)),
            *(random_dense_task(seed) for seed in range(20)),
        ),
    )


SUITES = {"four_rooms_v1": four_rooms_v1}


def task_suite(name: str) -> TaskSuite:
    try:
        return SUITES[name]()
    except KeyError:
        raise ValueError(f"unknown task suite {name!r}; available: {', '.join(SUITES)}") from None
