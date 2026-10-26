"""Compare deterministic policies with the exact optimal goal-reaching policy.

Works on any validated ``TabularDynamics``; nothing here knows about a
specific environment, a run directory or a file format.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass

import numpy as np

from aeris.analysis.tabular import (
    OptimalValues,
    PolicyDiagnostics,
    compare_deterministic_policy,
    value_iteration,
)
from aeris.domain.artifacts import DiscreteState, PolicySnapshot
from aeris.domain.reports import SnapshotReport, TabularPolicyReport
from aeris.environments.tabular import TabularDynamics, TabularMDP


@dataclass(frozen=True, slots=True)
class GoalOracle:
    """The exact goal-reaching task on known dynamics and its optimal values."""

    mdp: TabularMDP
    goal_index: int
    shortest_distances: np.ndarray
    optimal: OptimalValues

    @property
    def dynamics(self) -> TabularDynamics:
        return self.mdp.dynamics

    @property
    def states(self) -> tuple[DiscreteState, ...]:
        return self.mdp.dynamics.states

    @property
    def goal(self) -> DiscreteState:
        return self.states[self.goal_index]

    def optimal_policy(self) -> np.ndarray:
        """One optimal action per state (the first among ties)."""
        return np.argmax(self.optimal.optimal_actions, axis=1).astype(np.int64)

    def compare(self, actions: np.ndarray, *, max_steps: int) -> PolicyDiagnostics:
        return compare_deterministic_policy(
            actions,
            self.dynamics,
            self.optimal,
            goal_index=self.goal_index,
            max_steps=max_steps,
            shortest_distances=self.shortest_distances,
        )


def build_goal_oracle(
    dynamics: TabularDynamics, goal: DiscreteState, *, gamma: float
) -> GoalOracle:
    """Solve the goal-reaching task exactly (see ``TabularMDP.goal_reaching``)."""
    goal_index = dynamics.index_of(goal)
    mdp = TabularMDP.goal_reaching(dynamics, goal_index, gamma=gamma)
    return GoalOracle(
        mdp=mdp,
        goal_index=goal_index,
        shortest_distances=dynamics.shortest_path_lengths(goal_index),
        optimal=value_iteration(mdp),
    )


@dataclass(frozen=True, slots=True)
class SnapshotDiagnostics:
    actions: np.ndarray
    diagnostics: PolicyDiagnostics


@dataclass(frozen=True, slots=True)
class TabularPolicyAnalysis:
    """The report plus the arrays needed to plot it."""

    report: TabularPolicyReport
    oracle: GoalOracle
    snapshots: dict[str, SnapshotDiagnostics]


def analyze_policy_snapshots(
    oracle: GoalOracle,
    snapshots: Mapping[str, PolicySnapshot],
    *,
    max_steps: int,
    run_id: str,
    task_id: str,
    snapshot_names: Mapping[str, str],
) -> TabularPolicyAnalysis:
    """Score each labelled snapshot against the exact optimal policy.

    ``snapshot_names`` gives the artifact name recorded for each label in the report.
    """
    optimal_metrics = oracle.compare(oracle.optimal_policy(), max_steps=max_steps)
    results: dict[str, SnapshotDiagnostics] = {}
    entries: dict[str, SnapshotReport] = {}
    for label, snapshot in snapshots.items():
        actions = np.asarray(snapshot.actions_for(oracle.states), dtype=np.int64)
        diagnostics = oracle.compare(actions, max_steps=max_steps)
        results[label] = SnapshotDiagnostics(actions, diagnostics)
        entries[label] = SnapshotReport(
            policy_snapshot=snapshot_names[label],
            **asdict(diagnostics.summary()),
            success_gap_to_optimal=float(
                optimal_metrics.rollout_success_fraction - diagnostics.rollout_success_fraction
            ),
        )
    report = TabularPolicyReport(
        run_id=run_id,
        task_id=task_id,
        goal=oracle.goal,
        discount=oracle.mdp.gamma,
        rollout_horizon=max_steps,
        optimal_policy=optimal_metrics.summary(),
        maximum_shortest_path_steps=int(oracle.shortest_distances.max()),
        snapshots=entries,
    )
    return TabularPolicyAnalysis(report, oracle, results)
