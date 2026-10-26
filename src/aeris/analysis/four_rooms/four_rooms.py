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
from aeris.domain.spaces import DiscreteActionSpec
from aeris.environments.gridworld import FourRooms

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
        goal=(int(task.settings["goal_row"]), int(task.settings["goal_col"])),
        max_steps=int(experiment.evaluation.protocol.settings["max_steps"]),
    )


@dataclass(frozen=True, slots=True)
class FourRoomsOracle:
    positions: tuple[Position, ...]
    goal: Position
    goal_index: int
    next_state_indices: np.ndarray
    shortest_distances: np.ndarray
    optimal: OptimalValues

    def compare(self, actions: np.ndarray, *, max_steps: int) -> PolicyDiagnostics:
        return compare_deterministic_policy(
            actions,
            self.next_state_indices,
            self.optimal,
            goal_index=self.goal_index,
            max_steps=max_steps,
            shortest_distances=self.shortest_distances,
        )


def build_four_rooms_oracle(
    env: FourRooms, goal: Position, *, gamma: float = 0.99
) -> FourRoomsOracle:
    """Goal reward = 1 on ENTERING the goal; the goal then becomes terminal.

    This diagnostic reward is deliberately separate from the reward-free FB
    pretraining transitions. A reward is NOT repeatedly earned by waiting at
    the goal, nor assumed to have been present during training.
    """
    if goal not in env.positions:
        raise ValueError(f"invalid Four Rooms goal: {goal!r}")
    index = {pos: i for i, pos in enumerate(env.positions)}
    goal_index = index[goal]
    next_states = np.asarray(
        [
            [index[env.next_position(pos, a)] for a in range(env.num_actions)]
            for pos in env.positions
        ],
        dtype=np.int64,
    )
    rewards = (next_states == goal_index).astype(np.float64)
    rewards[goal_index] = 0.0
    done = np.zeros(len(env.positions), dtype=bool)
    done[goal_index] = True
    distances = np.asarray(
        [env.shortest_distance(pos, goal) for pos in env.positions], dtype=np.int64
    )
    optimal = value_iteration(next_states, rewards, done, gamma=gamma)
    return FourRoomsOracle(env.positions, goal, goal_index, next_states, distances, optimal)


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
        snapshot.require_action_space(DiscreteActionSpec(env.num_actions))
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
