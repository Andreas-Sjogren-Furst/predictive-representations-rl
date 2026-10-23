import numpy as np
import pytest

from aeris.contracts.environment import Environment
from aeris.environments import FourRooms


def test_environment_protocol_and_one_hot():
    env = FourRooms(start_position=(1, 1), max_episode_steps=3)
    assert isinstance(env, Environment)
    assert env.num_actions == 4
    obs = env.reset(seed=9).observation
    assert obs.shape == (121,)
    assert obs.dtype == np.float32
    assert env.position_of(obs) == (1, 1)
    assert obs.sum() == 1
    assert len(env.positions) == 68


def test_walls_and_doorways_are_consistent():
    env = FourRooms(start_position=(4, 2))
    env.reset(seed=0)
    assert env.step(1).info["position"] == (5, 2)  # doorway
    assert env.step(1).info["position"] == (6, 2)
    assert env.next_position((4, 4), 1) == (4, 4)  # wall
    assert env.shortest_distance((1, 1), (9, 9)) > 0
    assert env.shortest_distance((2, 2), (2, 2)) == 0


def test_deterministic_seeding_and_timeout():
    a = FourRooms(max_episode_steps=2)
    b = FourRooms(max_episode_steps=2)
    assert np.array_equal(a.reset(seed=17).observation, b.reset(seed=17).observation)
    assert a.step(0).truncated is False
    second = a.step(0)
    assert second.truncated is True and second.terminated is False
    assert second.reward is None
    with pytest.raises(RuntimeError):
        a.step(0)
    with pytest.raises(ValueError):
        a.next_position((1, 1), 7)


def test_exact_coverage_and_one_hot_validation():
    env = FourRooms()
    transitions = tuple(env.all_transitions())
    assert len(transitions) == len(env.positions) * 4
    assert all(t.reward is None for t in transitions)
    with pytest.raises(ValueError):
        env.position_of(np.zeros(121))
    with pytest.raises(ValueError):
        env.observation_for((5, 5))
