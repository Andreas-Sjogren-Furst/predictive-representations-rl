"""FB representation diagnostics against an exact tabular model.

Intentionally NOT a universal analysis: only FB exposes F(s, a, z) and B(s).
Inputs are an already loaded agent, the environment's exact model and a goal
oracle; nothing here reads files or checks whether a run is eligible.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from aeris.agents.fb.reports import FBCheckpointReport
from aeris.agents.fb.representations import FBRepresentations
from aeris.analysis.goal_policy import GoalOracle
from aeris.analysis.tabular_occupancy import centered_pearson, exact_discounted_occupancy
from aeris.contracts.environment import TabularModel
from aeris.domain.artifacts import DiscreteState, PolicySnapshot
from aeris.environments.tabular import TabularDynamics

if TYPE_CHECKING:
    from aeris.agents.fb.agent import DiscreteFBAgent


def _effective_rank(matrix: np.ndarray) -> float:
    """Entropy effective rank of centered state features; 0 if constant."""
    _, singular, _ = np.linalg.svd(matrix - matrix.mean(axis=0, keepdims=True), full_matrices=False)
    eig = singular**2
    if eig.sum() <= 1e-24:
        return 0.0
    p = eig[eig > 0] / eig.sum()
    return float(np.exp(-(p * np.log(p)).sum()))


def diagnose_fb_representation(
    agent: DiscreteFBAgent,
    model: TabularModel,
    oracle: GoalOracle,
    *,
    max_steps: int,
    run_id: str,
    checkpoint: str,
    final_snapshot: PolicySnapshot | None = None,
) -> FBCheckpointReport:
    """Diagnose representation occupancy and policy action ranking for the oracle's goal.

    The occupancy comparison assumes training data uniform over state-action
    pairs (density 1/S per state), which the caller must have verified. Does not
    claim numerical equivalence to the original paper or to terminal
    goal-reaching reward evaluation.
    """
    states = oracle.states
    n, num_actions = oracle.dynamics.num_states, oracle.dynamics.num_actions
    obs = np.stack([model.observation_for(state) for state in states])
    goal_z = agent.goal_condition(model.observation_for(oracle.goal))
    f = agent.forward_map(obs, goal_z).cpu().numpy()  # S, d, A
    b = agent.backward_features(obs).cpu().numpy()  # S, d
    policy = agent.target_policy(obs, goal_z).cpu().numpy()  # S, A
    zs = np.broadcast_to(goal_z.cpu().numpy(), (n, agent.config.representation_dim))
    learned_q = np.einsum("sda,sd->sa", f, zs)
    predicted_occupancy_density = np.einsum("sda,td->sat", f, b)
    if not np.isfinite(learned_q).all() or not np.isfinite(predicted_occupancy_density).all():
        raise ValueError("nonfinite network predictions")

    # Uniform data over (state, action) means rho(state) = 1/S.
    expected_occupancy_density = n * exact_discounted_occupancy(
        oracle.dynamics, policy, gamma=agent.config.gamma
    )
    non_goal = np.arange(n) != oracle.goal_index
    a = np.argmax(learned_q, axis=1)
    # Compare relative action advantages, not absolute Q magnitudes. The FB
    # objective has density/normalization conventions unlike terminal returns.
    actual_goal = expected_occupancy_density[:, :, oracle.goal_index]
    centered_learned = learned_q - learned_q.mean(axis=1, keepdims=True)
    centered_exact = actual_goal - actual_goal.mean(axis=1, keepdims=True)
    best_exact = np.max(actual_goal, axis=1)
    # The exact policy is induced by target F, not by the greedy online F.
    # This agreement tests ONE policy-improvement step against the induced pi.
    agreement = actual_goal[np.arange(n), a] >= best_exact - 1e-8
    top_two = np.partition(learned_q, -2, axis=1)[:, -2:]
    margin = top_two[:, 1] - top_two[:, 0]
    entropy = -np.sum(policy * np.log(np.maximum(policy, 1e-300)), axis=1)
    first_hit = oracle.compare(a.astype(np.int64), max_steps=max_steps)
    snapshot_matches = (
        None
        if final_snapshot is None
        else bool(np.array_equal(final_snapshot.actions_for(states), a))
    )
    return FBCheckpointReport(
        run_id=run_id,
        checkpoint=checkpoint,
        goal=oracle.goal,
        number_of_states=n,
        gradient_updates=agent.updates,
        gamma=agent.config.gamma,
        policy_temperature=agent.config.temperature,
        backward_effective_rank=_effective_rank(b),
        backward_goal_norm=float(np.linalg.norm(b[oracle.goal_index])),
        target_policy_mean_entropy_nats=float(entropy.mean()),
        target_policy_entropy_fraction_of_uniform=float(entropy.mean() / math.log(num_actions)),
        learned_action_spread_mean=float(np.std(learned_q[non_goal], axis=1).mean()),
        learned_action_margin_mean=float(margin[non_goal].mean()),
        state_positions=states,
        state_action_improvement_agreement=tuple(bool(item) for item in agreement),
        learned_action_counts_non_goal=tuple(
            int(np.sum(a[non_goal] == action)) for action in range(num_actions)
        ),
        exact_occupancy_correlation=centered_pearson(
            predicted_occupancy_density[non_goal], expected_occupancy_density[non_goal]
        ),
        goal_action_advantage_correlation=centered_pearson(
            centered_learned[non_goal], centered_exact[non_goal]
        ),
        goal_action_agreement_with_exact_policy_improvement=float(agreement[non_goal].mean()),
        first_hit_goal_performance=first_hit.summary(),
        final_snapshot_matches_checkpoint=snapshot_matches,
    )


def compute_fb_representations(
    agent: DiscreteFBAgent,
    model: TabularModel,
    dynamics: TabularDynamics,
    goals: Sequence[DiscreteState],
    *,
    state_density: np.ndarray,
    run_id: str,
    checkpoint: str,
) -> FBRepresentations:
    """B for every state and, per goal z_g = B(goal), F, the target policy and the exact density.

    ``state_density`` is rho, the frequency of each state among the training
    observations FB was compared against. FB's F·B estimates the occupancy
    density m(s, a, t) / rho(t), so the exact reference is divided by the same
    rho (NaN where a state never occurs in the data).
    """
    states = dynamics.states
    obs = np.stack([model.observation_for(state) for state in states])
    rho = np.asarray(state_density, dtype=np.float64)
    inverse_rho = np.divide(1.0, rho, out=np.full_like(rho, np.nan), where=rho > 0)
    conditions, forward, policies, exact = [], [], [], []
    for goal in goals:
        z = agent.goal_condition(model.observation_for(goal))
        policy = agent.target_policy(obs, z).cpu().numpy()
        conditions.append(z.cpu().numpy())
        forward.append(agent.forward_map(obs, z).cpu().numpy())
        policies.append(policy)
        occupancy = exact_discounted_occupancy(dynamics, policy, gamma=agent.config.gamma)
        exact.append(occupancy * inverse_rho[None, None, :])
    return FBRepresentations(
        run_id=run_id,
        checkpoint=checkpoint,
        gamma=agent.config.gamma,
        states=states,
        goals=tuple(tuple(goal) for goal in goals),
        backward=agent.backward_features(obs).cpu().numpy().astype(np.float32),
        goal_conditions=np.stack(conditions).astype(np.float32),
        forward=np.stack(forward).astype(np.float32),
        target_policy=np.stack(policies).astype(np.float32),
        state_density=rho.astype(np.float32),
        exact_density=np.stack(exact).astype(np.float32),
    )
