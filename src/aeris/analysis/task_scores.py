"""Exact scores of deterministic policies on arbitrary state-reward tasks.

Objective (the one FB's inferred z optimizes): the discounted sum of state
rewards, counting the current state, under a continuing policy,

    V^pi(s) = r(s) + gamma * V^pi(next(s, pi(s))).

Every value is computed exactly from the known dynamics, so scores carry no
sampling noise. Start states are weighted uniformly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aeris.analysis.tabular import OptimalValues, first_hit_steps, value_iteration
from aeris.environments.tabular import TabularDynamics, TabularMDP


@dataclass(frozen=True, slots=True)
class TaskBaseline:
    """Agent-independent references for one task: compute once, score many policies."""

    optimal_value: float  # mean V* over start states
    random_value: float  # mean V of the uniformly random policy


@dataclass(frozen=True, slots=True)
class TaskScore:
    normalized_score: float | None  # (V^pi - V^rand) / (V* - V^rand); None if V* = V^rand
    policy_value: float  # mean over start states
    optimal_value: float
    random_value: float
    goal_success: float | None  # goal tasks: fraction of non-goal starts reaching the goal in time


def _transition_matrix(dynamics: TabularDynamics, policy: np.ndarray) -> np.ndarray:
    """P_pi[s, t] for action probabilities ``policy`` [S, A]."""
    s = dynamics.num_states
    matrix = np.zeros((s, s))
    for action in range(dynamics.num_actions):
        np.add.at(matrix, (np.arange(s), dynamics.next_states[:, action]), policy[:, action])
    return matrix


def policy_values(
    dynamics: TabularDynamics, policy: np.ndarray, rewards: np.ndarray, *, gamma: float
) -> np.ndarray:
    """Exact V^pi for state rewards under action probabilities ``policy`` [S, A]."""
    transition = _transition_matrix(dynamics, policy)
    return np.linalg.solve(np.eye(dynamics.num_states) - gamma * transition, rewards)


def optimal_solution(
    dynamics: TabularDynamics, rewards: np.ndarray, *, gamma: float
) -> OptimalValues:
    """Exact V*, Q* and the optimal actions for state rewards, continuing (no terminal states)."""
    rewards = np.asarray(rewards, dtype=np.float64)
    mdp = TabularMDP(
        dynamics,
        np.repeat(rewards[:, None], dynamics.num_actions, axis=1),
        np.zeros(dynamics.num_states, dtype=bool),
        gamma,
    )
    return value_iteration(mdp)


def optimal_values(dynamics: TabularDynamics, rewards: np.ndarray, *, gamma: float) -> np.ndarray:
    """Exact V* for state rewards, continuing (no terminal states)."""
    return optimal_solution(dynamics, rewards, gamma=gamma).values


def deterministic_policy(actions: np.ndarray, num_actions: int) -> np.ndarray:
    policy = np.zeros((len(actions), num_actions))
    policy[np.arange(len(actions)), actions] = 1.0
    return policy


def task_baseline(dynamics: TabularDynamics, rewards: np.ndarray, *, gamma: float) -> TaskBaseline:
    rewards = np.asarray(rewards, dtype=np.float64)
    uniform = np.full((dynamics.num_states, dynamics.num_actions), 1.0 / dynamics.num_actions)
    return TaskBaseline(
        optimal_value=float(optimal_values(dynamics, rewards, gamma=gamma).mean()),
        random_value=float(policy_values(dynamics, uniform, rewards, gamma=gamma).mean()),
    )


def score_policy(
    dynamics: TabularDynamics,
    actions: np.ndarray,
    rewards: np.ndarray,
    baseline: TaskBaseline,
    *,
    gamma: float,
    goal_index: int | None = None,
    max_steps: int = 60,
) -> TaskScore:
    """Score one deterministic action per state on one reward function."""
    rewards = np.asarray(rewards, dtype=np.float64)
    policy = deterministic_policy(actions, dynamics.num_actions)
    value = float(policy_values(dynamics, policy, rewards, gamma=gamma).mean())
    gap = baseline.optimal_value - baseline.random_value
    success = None
    if goal_index is not None:
        steps = first_hit_steps(actions, dynamics, goal_index=goal_index, max_steps=max_steps)
        success = float(np.mean(np.delete(steps, goal_index) >= 0))
    return TaskScore(
        normalized_score=None if gap <= 1e-12 else (value - baseline.random_value) / gap,
        policy_value=value,
        optimal_value=baseline.optimal_value,
        random_value=baseline.random_value,
        goal_success=success,
    )


def start_scores(
    dynamics: TabularDynamics, actions: np.ndarray, rewards: np.ndarray, *, gamma: float
) -> np.ndarray:
    """The normalized score from each start state alone; NaN where V*(s) = V^rand(s).

    ``(V^pi(s) - V^rand(s)) / (V*(s) - V^rand(s))``: 1 is optimal from that start,
    0 no better than acting uniformly at random. ``score_policy`` normalizes the
    means over starts instead, so its score is not the mean of these.
    """
    rewards = np.asarray(rewards, dtype=np.float64)
    uniform = np.full((dynamics.num_states, dynamics.num_actions), 1.0 / dynamics.num_actions)
    policy = deterministic_policy(np.asarray(actions), dynamics.num_actions)
    value = policy_values(dynamics, policy, rewards, gamma=gamma)
    best = optimal_values(dynamics, rewards, gamma=gamma)
    random = policy_values(dynamics, uniform, rewards, gamma=gamma)
    gap = best - random
    scores = np.full(dynamics.num_states, np.nan)
    np.divide(value - random, gap, out=scores, where=gap > 1e-12)
    return scores
