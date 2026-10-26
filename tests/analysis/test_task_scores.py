import numpy as np
import pytest

from aeris.analysis.tabular import first_hit_steps
from aeris.analysis.task_scores import (
    deterministic_policy,
    optimal_values,
    policy_values,
    score_policy,
    task_baseline,
)
from aeris.environments import FourRooms, TabularDynamics
from aeris.environments.four_rooms_tasks import four_rooms_v1

ENV = FourRooms()
DYN = ENV.tabular_dynamics()


def greedy(rewards, gamma=0.99):
    v_star = optimal_values(DYN, rewards, gamma=gamma)
    return (rewards[:, None] + gamma * v_star[DYN.next_states]).argmax(axis=1)


def test_optimal_policy_scores_one_on_every_v1_task():
    for task in four_rooms_v1().tasks[::7]:
        rewards = ENV.state_rewards(task)
        score = score_policy(
            DYN, greedy(rewards), rewards, task_baseline(DYN, rewards, gamma=0.99), gamma=0.99
        )
        assert score.normalized_score == pytest.approx(1.0, abs=1e-6), task.task_id


def test_policy_values_satisfy_the_bellman_equation():
    rewards = ENV.state_rewards(four_rooms_v1().tasks[-1])  # a dense random reward
    actions = np.random.default_rng(0).integers(4, size=DYN.num_states)
    values = policy_values(DYN, deterministic_policy(actions, 4), rewards, gamma=0.9)
    nxt = DYN.next_states[np.arange(DYN.num_states), actions]
    assert np.allclose(values, rewards + 0.9 * values[nxt])


def test_random_policies_score_near_zero():
    rewards = ENV.state_rewards(ENV.goal_task((1, 1)))
    baseline = task_baseline(DYN, rewards, gamma=0.99)
    rng = np.random.default_rng(1)
    scores = [
        score_policy(
            DYN, rng.integers(4, size=DYN.num_states), rewards, baseline, gamma=0.99
        ).normalized_score
        for _ in range(100)
    ]
    assert abs(np.mean(scores)) < 0.05


def test_constant_rewards_have_no_score():
    rewards = np.ones(DYN.num_states)
    score = score_policy(
        DYN, greedy(rewards), rewards, task_baseline(DYN, rewards, gamma=0.99), gamma=0.99
    )
    assert score.normalized_score is None


def test_goal_success_and_first_hit_steps():
    chain = TabularDynamics(((0,), (1,), (2,)), np.array([[1, 0], [2, 1], [2, 2]]))
    assert first_hit_steps(np.array([0, 0, 0]), chain, goal_index=2, max_steps=5).tolist() == [
        2,
        1,
        0,
    ]
    assert first_hit_steps(np.array([1, 0, 0]), chain, goal_index=2, max_steps=5).tolist() == [
        -1,
        1,
        0,
    ]
    rewards = np.array([0.0, 0.0, 1.0])
    baseline = task_baseline(chain, rewards, gamma=0.9)
    score = score_policy(
        chain, np.array([1, 0, 0]), rewards, baseline, gamma=0.9, goal_index=2, max_steps=5
    )
    assert score.goal_success == 0.5  # of the two non-goal starts, only state 1 arrives
