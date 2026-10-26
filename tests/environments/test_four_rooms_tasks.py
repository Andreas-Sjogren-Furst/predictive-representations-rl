import numpy as np
import pytest

from aeris.domain import TaskSpec, TaskSuite, task_family
from aeris.environments import FourRooms
from aeris.environments.four_rooms import ROOMS, room_of
from aeris.environments.four_rooms_tasks import (
    distance_task,
    four_rooms_v1,
    random_dense_task,
    random_sparse_task,
    room_task,
    task_suite,
)

ENV = FourRooms()


def test_v1_suite_composition_is_stable():
    suite = four_rooms_v1()
    counts = {f: sum(task_family(t) == f for t in suite.tasks) for f in suite.families}
    assert counts == {"goal": 68, "room": 4, "distance": 8, "random_sparse": 20, "random_dense": 20}
    assert TaskSuite.from_dict(suite.to_dict()) == suite
    assert task_suite("four_rooms_v1") == suite
    with pytest.raises(ValueError, match="unknown task suite"):
        task_suite("nope")


def test_rooms_partition_the_cells_apart_from_doorways():
    sizes = {room: sum(room_of(p) == room for p in ENV.positions) for room in (*ROOMS, "doorway")}
    assert sizes == {
        "top-left": 16,
        "top-right": 16,
        "bottom-left": 16,
        "bottom-right": 16,
        "doorway": 4,
    }
    rewards = ENV.state_rewards(room_task("bottom-right"))
    assert rewards.sum() == 16 and rewards[ENV.positions.index((9, 9))] == 1


def test_goal_and_distance_rewards():
    goal = ENV.state_rewards(ENV.goal_task((9, 9)))
    assert goal.sum() == 1 and goal[ENV.positions.index((9, 9))] == 1
    legacy = TaskSpec("reach_9_9", {"goal_row": 9, "goal_col": 9})  # runs saved before task types
    assert np.array_equal(ENV.state_rewards(legacy), goal)
    distance = ENV.state_rewards(distance_task((9, 9)))
    assert distance.max() == 0 and distance.min() == -1
    assert distance[ENV.positions.index((9, 9))] == 0


def test_random_tasks_are_reproducible_and_differ_by_seed():
    assert np.array_equal(
        ENV.state_rewards(random_sparse_task(3)), ENV.state_rewards(random_sparse_task(3))
    )
    assert not np.array_equal(
        ENV.state_rewards(random_dense_task(1)), ENV.state_rewards(random_dense_task(2))
    )
    assert ENV.state_rewards(random_sparse_task(0, cells=5)).sum() == 5


def test_invalid_tasks_are_rejected():
    with pytest.raises(ValueError, match="unknown Four Rooms task type"):
        ENV.state_rewards(TaskSpec("x", {"type": "teleport"}))
    with pytest.raises(ValueError, match="room must be one of"):
        ENV.state_rewards(TaskSpec("x", {"type": "room", "room": "attic"}))
    with pytest.raises(ValueError, match="no single goal"):
        ENV.goal_from_task(room_task("top-left"))
