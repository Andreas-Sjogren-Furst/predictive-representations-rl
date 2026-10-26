"""FB-specific validation of one trusted, locally saved Four Rooms checkpoint.

This is intentionally NOT a universal analysis capability: only FB can expose
F(s,a,z) and B(s). Numerical tabular occupancy lives in a generic module.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from aeris.agents.fb.reports import FBCheckpointReport
from aeris.analysis.four_rooms.four_rooms import build_four_rooms_oracle, load_four_rooms_run
from aeris.analysis.tabular_occupancy import centered_pearson, exact_discounted_occupancy
from aeris.data.artifacts import TransitionFile
from aeris.data.run_directory import RunDirectory
from aeris.domain.spaces import DiscreteActionSpec
from aeris.environments import FourRooms


def _effective_rank(matrix: np.ndarray) -> float:
    """Entropy effective rank of centered state features; 0 if constant."""
    _, singular, _ = np.linalg.svd(matrix - matrix.mean(axis=0, keepdims=True), full_matrices=False)
    eig = singular**2
    if eig.sum() <= 1e-24:
        return 0.0
    p = eig[eig > 0] / eig.sum()
    return float(np.exp(-(p * np.log(p)).sum()))


def _require_full_coverage(dataset: TransitionFile, env: FourRooms) -> None:
    """Verify uniform state-action coverage instead of trusting a dataset label."""
    needed = len(env.positions) * env.num_actions
    if len(dataset) != needed or dataset.observations.shape[1] != env.observation_dim:
        raise ValueError("oracle dataset must contain exactly one sample per state-action")
    observed = set()
    for transition in dataset.iter_transitions():
        pos = env.position_of(transition.observation)
        if not 0 <= transition.action < env.num_actions:
            raise ValueError("oracle dataset contains invalid action")
        if env.position_of(transition.next_observation) != env.next_position(
            pos, transition.action
        ):
            raise ValueError("oracle dataset contains incorrect transition")
        observed.add((pos, transition.action))
    if len(observed) != needed:
        raise ValueError("oracle dataset contains duplicate or missing state-actions")


def analyze_fb_checkpoint(run_dir: str | Path | RunDirectory) -> FBCheckpointReport:
    """Diagnose representation occupancy and policy action ranking for a goal.

    Only applies the uniform-state-density comparison to the explicitly saved
    full-coverage oracle dataset. Does not claim numerical equivalence to the
    original paper or to terminal goal-reaching reward evaluation. The report
    is written to ``analysis/fb_checkpoint_v1.json``.
    """
    from aeris.agents.fb.checkpoint import load_agent  # needs torch

    run = load_four_rooms_run(run_dir)
    dataset_spec = run.manifest.experiment.dataset
    settings = dataset_spec.settings if dataset_spec is not None else {}
    if settings.get("origin") != "oracle":
        raise ValueError("occupancy comparison requires the saved uniform-coverage oracle dataset")
    env = FourRooms()
    _require_full_coverage(
        TransitionFile(run.store.dataset_path, expected_sha256=settings.get("sha256")), env
    )
    agent = load_agent(run.store.checkpoint_path)
    if (
        agent.config.observation_dim != env.observation_dim
        or agent.config.num_actions != env.num_actions
    ):
        raise ValueError("checkpoint is incompatible with Four Rooms")
    oracle = build_four_rooms_oracle(env, run.goal, gamma=agent.config.gamma)
    n = len(oracle.positions)
    obs = np.stack([env.observation_for(pos) for pos in oracle.positions])
    goal_z = agent.goal_condition(env.observation_for(run.goal))
    f = agent.forward_map(obs, goal_z).cpu().numpy()  # S, d, A
    b = agent.backward_features(obs).cpu().numpy()  # S, d
    policy = agent.target_policy(obs, goal_z).cpu().numpy()  # S, A
    zs = np.broadcast_to(goal_z.cpu().numpy(), (n, agent.config.representation_dim))
    learned_q = np.einsum("sda,sd->sa", f, zs)
    predicted_occupancy_density = np.einsum("sda,td->sat", f, b)

    if not np.isfinite(learned_q).all() or not np.isfinite(predicted_occupancy_density).all():
        raise ValueError("nonfinite network predictions")
    # Oracle dataset is uniform over (state, action), hence rho(state)=1/S.
    expected_occupancy_density = n * exact_discounted_occupancy(
        oracle.next_state_indices, policy, gamma=agent.config.gamma
    )
    non_goal = np.arange(n) != oracle.goal_index
    snapshot = run.store.read_policy_snapshot("final")
    if snapshot is not None:
        snapshot.require_action_space(DiscreteActionSpec(env.num_actions))
        snapshot_actions = np.asarray(snapshot.actions_for(oracle.positions))
        if not np.array_equal(snapshot_actions, np.argmax(learned_q, axis=1)):
            raise ValueError("saved final policy snapshot disagrees with the checkpoint")
    # Compare relative action advantages, not absolute Q magnitudes. The FB
    # objective has density/normalization conventions unlike terminal returns.
    actual_goal = expected_occupancy_density[:, :, oracle.goal_index]
    centered_learned = learned_q - learned_q.mean(axis=1, keepdims=True)
    centered_exact = actual_goal - actual_goal.mean(axis=1, keepdims=True)
    a = np.argmax(learned_q, axis=1)
    best_exact = np.max(actual_goal, axis=1)
    agreement = actual_goal[np.arange(n), a] >= best_exact - 1e-8
    # The exact policy is induced by target F, not by the greedy online F.
    # This agreement tests ONE policy-improvement step against the induced pi.
    top_two = np.partition(learned_q, -2, axis=1)[:, -2:]
    margin = top_two[:, 1] - top_two[:, 0]
    entropy = -np.sum(policy * np.log(np.maximum(policy, 1e-300)), axis=1)
    first_hit = oracle.compare(a.astype(np.int64), max_steps=run.max_steps)
    report = FBCheckpointReport(
        run_id=run.manifest.run_id,
        checkpoint=run.store.checkpoint_path.name,
        goal=run.goal,
        number_of_states=n,
        gradient_updates=agent.updates,
        gamma=agent.config.gamma,
        policy_temperature=agent.config.temperature,
        backward_effective_rank=_effective_rank(b),
        backward_goal_norm=float(np.linalg.norm(b[oracle.goal_index])),
        target_policy_mean_entropy_nats=float(entropy.mean()),
        target_policy_entropy_fraction_of_uniform=float(entropy.mean() / math.log(env.num_actions)),
        learned_action_spread_mean=float(np.std(learned_q[non_goal], axis=1).mean()),
        learned_action_margin_mean=float(margin[non_goal].mean()),
        state_positions=oracle.positions,
        state_action_improvement_agreement=tuple(bool(item) for item in agreement),
        learned_action_counts_non_goal=tuple(
            int(np.sum(a[non_goal] == action)) for action in range(env.num_actions)
        ),
        exact_occupancy_correlation=centered_pearson(
            predicted_occupancy_density[non_goal], expected_occupancy_density[non_goal]
        ),
        goal_action_advantage_correlation=centered_pearson(
            centered_learned[non_goal], centered_exact[non_goal]
        ),
        goal_action_agreement_with_exact_policy_improvement=float(agreement[non_goal].mean()),
        first_hit_goal_performance=first_hit.summary(),
    )
    run.store.write_report(report)
    return report
