"""Generic task-conditioned goal evaluation; task evidence is handled upstream."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from aeris.contracts.agent import Agent
from aeris.contracts.environment import Environment
from aeris.domain.agent import ActionRequest
from aeris.domain.results import EvaluationRecord, MetricValue
from aeris.domain.task import TaskEvidenceRef
from aeris.domain.trajectory import EvaluationTrajectory, TrajectoryStep
from aeris.domain.validation import require_int


def evaluate_goal_policy(
    agent: Agent,
    env: Environment,
    *,
    condition: Any,
    is_goal: Callable[[Any], bool],
    episodes: int,
    max_steps: int,
    run_id: str,
    task_id: str,
    checkpoint_id: str,
    seed: int,
    protocol_id: str = "goal_reaching_v1",
    on_trajectory: Callable[[EvaluationTrajectory], None] | None = None,
    max_trajectories: int | None = None,
    inference_evidence: TaskEvidenceRef | None = None,
) -> EvaluationRecord:
    """Evaluate a fixed policy condition, optionally emitting selected episodes.

    ``inference_evidence`` names the evidence a task condition was inferred
    from, so the record shows which zero-shot protocol produced it.
    The callback only observes completed evaluation episodes; it never changes
    the policy and is never used as task-inference evidence. For large visual
    observations, avoid JSONL capture (use a binary artifact adapter later).
    """
    require_int("episodes", episodes, minimum=1)
    require_int("max_steps", max_steps, minimum=1)
    if max_trajectories is not None:
        require_int("max_trajectories", max_trajectories)
    successes = 0
    lengths: list[int] = []
    for episode in range(episodes):
        episode_seed = seed + episode
        obs = env.reset(seed=episode_seed).observation
        initial_obs = obs
        policy_state = None
        success = bool(is_goal(obs))
        length = 0
        capture = on_trajectory is not None and (
            max_trajectories is None or episode < max_trajectories
        )
        steps: list[TrajectoryStep] = []
        while not success and length < max_steps:
            result = agent.act(
                ActionRequest(
                    observation=obs,
                    condition=condition,
                    policy_state=policy_state,
                    deterministic=True,
                )
            )
            transition = env.step(result.action)
            nxt = transition.observation
            reached = bool(is_goal(nxt))
            if capture:
                steps.append(
                    TrajectoryStep(
                        observation=obs,
                        action=result.action,
                        next_observation=nxt,
                        reward=transition.reward,
                        terminated=transition.terminated,
                        truncated=transition.truncated,
                        goal_reached=reached,
                    )
                )
            obs = nxt
            policy_state = result.next_policy_state
            length += 1
            success = reached
            if transition.terminated or transition.truncated:
                break
        successes += int(success)
        lengths.append(length)
        if capture and on_trajectory is not None:
            on_trajectory(
                EvaluationTrajectory(
                    run_id=run_id,
                    task_id=task_id,
                    checkpoint_id=checkpoint_id,
                    protocol_id=protocol_id,
                    seed=episode_seed,
                    episode_index=episode,
                    initial_observation=initial_obs,
                    steps=tuple(steps),
                    success=success,
                )
            )
    return EvaluationRecord(
        run_id=run_id,
        task_id=task_id,
        checkpoint_id=checkpoint_id,
        protocol_id=protocol_id,
        seed=seed,
        episode_count=episodes,
        inference_evidence=inference_evidence,
        metrics=(
            MetricValue(
                name="success_rate", value=successes / episodes, unit="fraction", aggregation="mean"
            ),
            MetricValue(
                name="mean_episode_length",
                value=sum(lengths) / episodes,
                unit="steps",
                aggregation="mean",
            ),
        ),
    )
