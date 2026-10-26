"""Four Rooms exact tabular model and saved-policy diagnostics.

This is deliberately environment-specific. The generic value iteration and
policy-comparison functions in tabular.py have no dependency on FourRooms.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from aeris.analysis.tabular import (
    OptimalValues,
    PolicyDiagnostics,
    compare_deterministic_policy,
    value_iteration,
)
from aeris.data.run_directory import RunDirectory
from aeris.domain.reports import SnapshotReport, TabularPolicyReport
from aeris.domain.results import RunManifest
from aeris.environments.four_rooms import FourRooms
from aeris.environments.tabular import TabularDynamics, TabularMDP

Position = tuple[int, int]
SNAPSHOT_PHASES = ("initial", "final")


@dataclass(frozen=True, slots=True)
class FourRoomsRun:
    """A completed single-goal Four Rooms run and the values analyses need from it."""

    store: RunDirectory
    manifest: RunManifest
    task_id: str
    goal: Position
    max_steps: int


def load_four_rooms_run(run_dir: str | Path | RunDirectory) -> FourRoomsRun:
    store = run_dir if isinstance(run_dir, RunDirectory) else RunDirectory(run_dir)
    manifest = store.read_manifest()
    if manifest.status != "completed":
        raise ValueError("run must be completed")
    experiment = manifest.experiment
    if experiment.environment is None or experiment.environment.name != "four_rooms":
        raise ValueError("only Four Rooms runs are supported")
    if experiment.evaluation is None or len(experiment.evaluation.targets) != 1:
        raise ValueError("exactly one evaluation target is required")
    task = experiment.evaluation.targets[0].task
    return FourRoomsRun(
        store=store,
        manifest=manifest,
        task_id=task.task_id,
        goal=FourRooms().goal_from_task(task),
        max_steps=int(experiment.evaluation.protocol.settings["max_steps"]),
    )


@dataclass(frozen=True, slots=True)
class FourRoomsOracle:
    """Exact goal-reaching model of Four Rooms and its optimal values."""

    mdp: TabularMDP
    goal: Position
    goal_index: int
    shortest_distances: np.ndarray
    optimal: OptimalValues

    @property
    def dynamics(self) -> TabularDynamics:
        return self.mdp.dynamics

    @property
    def positions(self) -> tuple[Position, ...]:
        return self.mdp.dynamics.states

    @property
    def next_state_indices(self) -> np.ndarray:
        return self.mdp.dynamics.next_states

    def compare(self, actions: np.ndarray, *, max_steps: int) -> PolicyDiagnostics:
        return compare_deterministic_policy(
            actions,
            self.dynamics,
            self.optimal,
            goal_index=self.goal_index,
            max_steps=max_steps,
            shortest_distances=self.shortest_distances,
        )


def build_four_rooms_oracle(
    env: FourRooms, goal: Position, *, gamma: float = 0.99
) -> FourRoomsOracle:
    """Solve the goal-reaching task exactly (see ``TabularMDP.goal_reaching``)."""
    dynamics = env.tabular_dynamics()
    goal_index = dynamics.index_of(goal)
    mdp = TabularMDP.goal_reaching(dynamics, goal_index, gamma=gamma)
    return FourRoomsOracle(
        mdp=mdp,
        goal=goal,
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
    """The saved report plus the arrays needed to plot it."""

    report: TabularPolicyReport
    oracle: FourRoomsOracle
    snapshots: dict[str, SnapshotDiagnostics]


def analyze_saved_four_rooms_run(
    run_dir: str | Path | RunDirectory, *, gamma: float = 0.99
) -> TabularPolicyAnalysis:
    """Compare saved policy snapshots with the exact optimal policy, without torch.

    The original experiment artifacts remain unchanged. The report is derived
    from saved policy snapshots, not from loading or evaluating the model, and
    is saved as ``analysis/tabular_policy_v1.json``.
    """
    run = load_four_rooms_run(run_dir)
    env = FourRooms()
    oracle = build_four_rooms_oracle(env, run.goal, gamma=gamma)
    optimal_policy = np.argmax(oracle.optimal.optimal_actions, axis=1).astype(np.int64)
    optimal_metrics = oracle.compare(optimal_policy, max_steps=run.max_steps)
    snapshots: dict[str, SnapshotDiagnostics] = {}
    snapshot_reports: dict[str, SnapshotReport] = {}
    for phase in SNAPSHOT_PHASES:
        snapshot = run.store.read_policy_snapshot(phase)
        if snapshot is None:
            continue
        snapshot.require_action_space(env.spec.action)
        actions = np.asarray(snapshot.actions_for(oracle.positions), dtype=np.int64)
        diagnostics = oracle.compare(actions, max_steps=run.max_steps)
        snapshots[phase] = SnapshotDiagnostics(actions, diagnostics)
        snapshot_reports[phase] = SnapshotReport(
            policy_snapshot=run.store.policy_snapshot_path(phase).name,
            **asdict(diagnostics.summary()),
            success_gap_to_optimal=float(
                optimal_metrics.rollout_success_fraction - diagnostics.rollout_success_fraction
            ),
        )
    if not snapshots:
        raise ValueError("no saved policy snapshots were found")
    report = TabularPolicyReport(
        run_id=run.manifest.run_id,
        task_id=run.task_id,
        goal=run.goal,
        discount=gamma,
        rollout_horizon=run.max_steps,
        optimal_policy=optimal_metrics.summary(),
        maximum_shortest_path_steps=int(oracle.shortest_distances.max()),
        snapshots=snapshot_reports,
    )
    run.store.write_report(report)
    return TabularPolicyAnalysis(report, oracle, snapshots)
