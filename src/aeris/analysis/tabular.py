"""Exact deterministic tabular MDP diagnostics, independent of the agent.

Inputs are validated ``TabularMDP`` / ``TabularDynamics`` models: discrete
states and actions, known deterministic transitions, rewards on transition,
absorbing terminals and 0 <= gamma < 1. Partially observed or stochastic
environments are out of scope.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aeris.domain.reports import PolicySummary
from aeris.environments.tabular import TabularDynamics, TabularMDP


@dataclass(frozen=True, slots=True)
class OptimalValues:
    values: np.ndarray  # [state]
    action_values: np.ndarray  # [state, action]
    optimal_actions: np.ndarray  # [state, action] bool, false at terminal states
    iterations: int


def value_iteration(
    mdp: TabularMDP,
    *,
    tolerance: float = 1e-11,
    max_iterations: int = 10_000,
    tie_tolerance: float = 1e-8,
) -> OptimalValues:
    """Solve V*(s)=max_a [r(s,a)+gamma V*(next(s,a))] in a deterministic MDP."""
    states = mdp.dynamics.next_states
    r, done, gamma = mdp.rewards, mdp.terminals, mdp.gamma
    values = np.zeros(mdp.dynamics.num_states, dtype=np.float64)
    for iteration in range(1, max_iterations + 1):  # noqa: B007 (reported after the loop)
        q = r + gamma * values[states]
        q[done] = 0.0
        updated = q.max(axis=1)
        converged = np.max(np.abs(updated - values)) <= tolerance
        values = updated
        if converged:
            break
    else:
        raise RuntimeError("value iteration did not converge")

    action_values = r + gamma * values[states]
    action_values[done] = 0.0
    is_optimal = action_values >= (action_values.max(axis=1, keepdims=True) - tie_tolerance)
    is_optimal[done] = False  # No action choice is needed once the goal is reached.
    return OptimalValues(values, action_values, is_optimal, iteration)


@dataclass(frozen=True, slots=True)
class PolicyDiagnostics:
    optimal_action_fraction: float
    mean_action_regret: float
    maximum_action_regret: float
    rollout_success_fraction: float
    mean_excess_steps_on_success: float | None
    optimal_by_state: np.ndarray
    regret_by_state: np.ndarray
    succeeds_by_state: np.ndarray
    rollout_steps_by_state: np.ndarray

    def summary(self) -> PolicySummary:
        return PolicySummary(
            optimal_action_fraction=self.optimal_action_fraction,
            mean_action_regret=self.mean_action_regret,
            maximum_action_regret=self.maximum_action_regret,
            rollout_success_fraction=self.rollout_success_fraction,
            mean_excess_steps_on_success=self.mean_excess_steps_on_success,
        )


def compare_deterministic_policy(
    actions: np.ndarray,
    dynamics: TabularDynamics,
    optimal: OptimalValues,
    *,
    goal_index: int,
    max_steps: int,
    shortest_distances: np.ndarray,
) -> PolicyDiagnostics:
    """Assess every non-goal start state with a fixed deterministic action map.

    Fractions weight all non-goal states uniformly, not by the evaluation reset
    distribution. This is an exact, model-based diagnostic, not sampled return.
    ``actions`` holds one valid action index per state of ``dynamics``.
    """
    nxt, act, distances = dynamics.next_states, np.asarray(actions), shortest_distances
    num_states = dynamics.num_states
    states = np.arange(num_states)
    non_goal = states != goal_index
    choice_q = optimal.action_values[states, act]
    q_gap = np.maximum(0.0, optimal.action_values.max(axis=1) - choice_q)
    optimal_mask = optimal.optimal_actions[states, act]
    succeeded = np.zeros(num_states, dtype=bool)
    lengths = np.full(num_states, -1, dtype=np.int64)
    succeeded[goal_index] = True
    lengths[goal_index] = 0
    for start in states[non_goal]:
        current = int(start)
        seen = {current}
        for step in range(1, max_steps + 1):
            current = int(nxt[current, act[current]])
            if current == goal_index:
                succeeded[start] = True
                lengths[start] = step
                break
            if current in seen:  # deterministic cycle: no future path to goal
                break
            seen.add(current)
    successful_starts = succeeded & non_goal
    mean_excess = (
        float(np.mean(lengths[successful_starts] - distances[successful_starts]))
        if np.any(successful_starts)
        else None
    )
    return PolicyDiagnostics(
        optimal_action_fraction=float(np.mean(optimal_mask[non_goal])),
        mean_action_regret=float(np.mean(q_gap[non_goal])),
        maximum_action_regret=float(np.max(q_gap[non_goal])),
        rollout_success_fraction=float(np.mean(succeeded[non_goal])),
        mean_excess_steps_on_success=mean_excess,
        optimal_by_state=optimal_mask,
        regret_by_state=q_gap,
        succeeds_by_state=succeeded,
        rollout_steps_by_state=lengths,
    )
