"""Zero-shot evaluation: infer a task condition from permitted evidence, then act."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from aeris.contracts.agent import ZeroShotAgent
from aeris.contracts.environment import Environment
from aeris.domain.results import EvaluationRecord
from aeris.domain.task import TaskEvidenceRef, TaskSpec
from aeris.experiments.evaluation import evaluate_goal_policy

EvidenceT = TypeVar("EvidenceT")


def evaluate_zero_shot(
    agent: ZeroShotAgent[EvidenceT],
    env: Environment,
    *,
    task: TaskSpec,
    evidence: EvidenceT,
    evidence_ref: TaskEvidenceRef,
    is_goal: Callable[[Any], bool],
    episodes: int,
    max_steps: int,
    run_id: str,
    checkpoint_id: str,
    seed: int,
    protocol_id: str = "zero_shot_goal_reaching_v1",
) -> EvaluationRecord:
    """Infer the task condition once from ``evidence``, then evaluate the fixed policy.

    The two phases are kept apart: inference sees only the supplied evidence
    (never evaluation rollouts), and evaluation never updates the agent. The
    record names the evidence through ``evidence_ref``.
    """
    condition = agent.infer_condition(task, evidence)
    return evaluate_goal_policy(
        agent,
        env,
        condition=condition,
        is_goal=is_goal,
        episodes=episodes,
        max_steps=max_steps,
        run_id=run_id,
        task_id=task.task_id,
        checkpoint_id=checkpoint_id,
        seed=seed,
        protocol_id=protocol_id,
        inference_evidence=evidence_ref,
    )
