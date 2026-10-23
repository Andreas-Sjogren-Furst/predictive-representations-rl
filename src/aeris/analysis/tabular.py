"""Exact deterministic tabular MDP diagnostics, independent of the agent.

This solver assumes discrete states/actions, known deterministic transitions,
rewards received on transition, absorbing terminal states, and 0 <= gamma < 1.
It does not claim to evaluate partially observed or stochastic environments.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aeris.domain.validation import require_int


@dataclass(frozen=True, slots=True)
class OptimalValues:
    values: np.ndarray  # [state]
    action_values: np.ndarray  # [state, action]
    optimal_actions: np.ndarray  # [state, action] bool, false at terminal states
    iterations: int


def value_iteration(
    next_states: np.ndarray,
    rewards: np.ndarray,
    terminals: np.ndarray,
    *,
    gamma: float = 0.99,
    tolerance: float = 1e-11,
    max_iterations: int = 10_000,
    tie_tolerance: float = 1e-8,
) -> OptimalValues:
    """Solve V*(s)=max_a [r(s,a)+gamma V*(next(s,a))] in a deterministic MDP."""
    states = np.asarray(next_states)
    r = np.asarray(rewards)
    done = np.asarray(terminals)
    if states.ndim != 2 or not states.size or states.shape[1] < 1 or states.dtype.kind not in "iu":
        raise ValueError("next_states must have shape [states, actions] with integer indices")
    num_states = states.shape[0]
    if np.any((states < 0) | (states >= num_states)):
        raise ValueError("next_states contains an out-of-range index")
    if r.shape != states.shape or r.dtype.kind not in "iuf" or not np.all(np.isfinite(r)):
        raise ValueError("rewards must be finite and match next_states shape")
    if done.shape != (num_states,) or done.dtype.kind != "b":
        raise ValueError("terminals must be a boolean vector of length num_states")
    if not np.isfinite(gamma) or not 0 <= gamma < 1:
        raise ValueError("gamma must lie in [0, 1)")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be positive and finite")
    if not np.isfinite(tie_tolerance) or tie_tolerance < 0:
        raise ValueError("tie_tolerance must be nonnegative and finite")
    require_int("max_iterations", max_iterations, minimum=1)

    r = r.astype(np.float64)
    values = np.zeros(num_states, dtype=np.float64)
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

    def summary(self) -> dict[str, float | None]:
        return {
            "optimal_action_fraction": self.optimal_action_fraction,
            "mean_action_regret": self.mean_action_regret,
            "maximum_action_regret": self.maximum_action_regret,
            "rollout_success_fraction": self.rollout_success_fraction,
            "mean_excess_steps_on_success": self.mean_excess_steps_on_success,
        }


def compare_deterministic_policy(
    actions: np.ndarray,
    next_states: np.ndarray,
    optimal: OptimalValues,
    *,
    goal_index: int,
    max_steps: int,
    shortest_distances: np.ndarray,
) -> PolicyDiagnostics:
    """Assess every non-goal start state with a fixed deterministic action map.

    Fractions weight all non-goal states uniformly, not by the evaluation reset
    distribution. This is an exact, model-based diagnostic, not sampled return.
    """
    nxt = np.asarray(next_states)
    act = np.asarray(actions)
    distances = np.asarray(shortest_distances)
    num_states, num_actions = nxt.shape
    if (
        act.shape != (num_states,)
        or act.dtype.kind not in "iu"
        or np.any((act < 0) | (act >= num_actions))
    ):
        raise ValueError("actions must be valid integer choices for every state")
    if distances.shape != (num_states,) or np.any(distances < 0):
        raise ValueError("shortest_distances must be finite nonnegative lengths")
    require_int("goal_index", goal_index)
    if goal_index >= num_states:
        raise ValueError("goal_index is invalid")
    require_int("max_steps", max_steps, minimum=1)
    if optimal.action_values.shape != (num_states, num_actions):
        raise ValueError("optimal values are incompatible with environment")

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
