import numpy as np
import pytest

from aeris.analysis.policy_evaluation import (
    EvaluatedPolicy,
    PolicyEvaluation,
    dataset_q,
    effective_horizon,
    exact_q,
)
from aeris.analysis.tabular_occupancy import exact_discounted_occupancy
from aeris.analysis.task_scores import optimal_solution
from aeris.environments import FourRooms
from conftest import synthetic_policy_evaluation

DYNAMICS = FourRooms().tabular_dynamics()
GOAL = DYNAMICS.index_of((9, 9))
REWARDS = np.eye(68)[GOAL]
UNIFORM_TABLE = np.full((68, 4), 0.25)


def optimal_table():
    best = optimal_solution(DYNAMICS, REWARDS, gamma=0.99)
    return best.optimal_actions / best.optimal_actions.sum(axis=1, keepdims=True), best


def test_exact_q_of_the_optimal_policy_is_q_star():
    table, best = optimal_table()
    assert exact_q(DYNAMICS, table, REWARDS, gamma=0.99) == pytest.approx(best.action_values)


def test_dataset_q_needs_every_pair_the_policy_can_reach():
    table, _ = optimal_table()
    full = np.ones((68, 4), dtype=bool)
    exact = exact_q(DYNAMICS, table, REWARDS, gamma=0.99)
    assert dataset_q(DYNAMICS, table, REWARDS, full, gamma=0.99) == pytest.approx(exact)
    no_left = full.copy()
    no_left[:, 2] = False
    uniform = dataset_q(DYNAMICS, UNIFORM_TABLE, REWARDS, no_left, gamma=0.99)
    assert np.isnan(uniform).all()  # the uniform policy always may go left
    partial = dataset_q(DYNAMICS, table, REWARDS, no_left, gamma=0.99)
    known = np.isfinite(partial)
    assert known.any() and not known[:, 2].any()
    assert partial[known] == pytest.approx(exact[known])  # what it knows is exact


def synthetic_evaluation(forward=None):
    return synthetic_policy_evaluation(forward)


def test_exact_successor_measure_gives_exact_q():
    # With B = I and uniform rho, F(s, a) = M(s, a, ·) / rho makes F·B·rho = M exactly.
    table = UNIFORM_TABLE
    occupancy = exact_discounted_occupancy(DYNAMICS, table, gamma=0.99)  # [S, A, S]
    forward = np.repeat((occupancy * 68).transpose(0, 2, 1)[None], 3, axis=0)
    evaluation = synthetic_evaluation(forward)
    assert evaluation.learned_q(0, REWARDS) == pytest.approx(
        exact_q(DYNAMICS, table, REWARDS, gamma=0.99)
    )


def test_policy_evaluation_round_trip():
    evaluation = synthetic_evaluation()
    again = PolicyEvaluation.from_arrays(*evaluation.to_arrays())
    assert again.policies == evaluation.policies and again.policy_index("uniform random") == 0
    assert again.covered.dtype == bool and np.isnan(again.losses[2]).all()
    assert not again.covered[:, 2].any()
    with pytest.raises(ValueError, match="tables has shape"):
        PolicyEvaluation(**{**vars(evaluation), "tables": np.ones((2, 68, 4))})
    with pytest.raises(ValueError, match="unknown policy kind"):
        EvaluatedPolicy("x", "greedy", None)


def test_effective_horizon_finds_a_truncated_q():
    table = UNIFORM_TABLE
    full = exact_q(DYNAMICS, table, REWARDS, gamma=0.99)
    horizon, error = effective_horizon(DYNAMICS, table, REWARDS, full, gamma=0.99, longest=2000)
    assert horizon > 1000 and error < 1e-3  # the infinite-horizon Q needs every step
    only_now = REWARDS[:, None] * np.ones((1, 4))  # the current reward alone
    assert effective_horizon(DYNAMICS, table, REWARDS, only_now, gamma=0.99) == (1, 0.0)
