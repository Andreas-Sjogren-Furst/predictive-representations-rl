"""Four Rooms run artifacts, exact tabular model and saved-policy diagnostics.

This is deliberately environment-specific. The generic value iteration and
policy-comparison functions in tabular.py have no dependency on FourRooms.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from aeris.analysis.tabular import (
    OptimalValues,
    PolicyDiagnostics,
    compare_deterministic_policy,
    value_iteration,
)
from aeris.contracts.agent import Agent
from aeris.domain.agent import ActionRequest
from aeris.environments.gridworld import FourRooms

Position = tuple[int, int]
SNAPSHOT_PHASES = ("initial", "final")


@dataclass(frozen=True, slots=True)
class FourRoomsRun:
    """A completed single-goal Four Rooms run directory and its manifest."""

    root: Path
    manifest: dict[str, Any]
    task_id: str | None
    goal: Position
    max_steps: int

    @property
    def experiment(self) -> dict[str, Any]:
        return self.manifest["experiment"]

    def write_report(self, name: str, report: dict[str, Any]) -> Path:
        path = self.root / "analysis" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        return path


def load_four_rooms_run(run_dir: str | Path) -> FourRoomsRun:
    root = Path(run_dir)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "completed":
        raise ValueError("run must be completed")
    experiment = manifest["experiment"]
    if experiment["environment"]["name"] != "four_rooms":
        raise ValueError("only Four Rooms runs are supported")
    targets = experiment["evaluation"]["targets"]
    if len(targets) != 1:
        raise ValueError("exactly one evaluation target is required")
    task = targets[0]["task"]
    settings = task["settings"]
    return FourRoomsRun(
        root=root,
        manifest=manifest,
        task_id=task.get("task_id"),
        goal=(int(settings["goal_row"]), int(settings["goal_col"])),
        max_steps=int(experiment["evaluation"]["protocol"]["settings"]["max_steps"]),
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


def build_four_rooms_oracle(env: FourRooms, goal: Position, *, gamma: float = 0.99) -> FourRoomsOracle:
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
        [[index[env.next_position(pos, a)] for a in range(env.num_actions)] for pos in env.positions],
        dtype=np.int64,
    )
    rewards = (next_states == goal_index).astype(np.float64)
    rewards[goal_index] = 0.0
    done = np.zeros(len(env.positions), dtype=bool)
    done[goal_index] = True
    distances = np.asarray([env.shortest_distance(pos, goal) for pos in env.positions], dtype=np.int64)
    optimal = value_iteration(next_states, rewards, done, gamma=gamma)
    return FourRoomsOracle(env.positions, goal, goal_index, next_states, distances, optimal)


def save_policy_snapshot(agent: Agent, env: FourRooms, condition: Any, output_file: str | Path) -> Path:
    """Record the deterministic action in every state, using only ``Agent.act``."""
    positions = np.asarray(env.positions, dtype=np.int64)
    actions = np.asarray(
        [
            agent.act(
                ActionRequest(
                    observation=env.observation_for(position), condition=condition, deterministic=True
                )
            ).action
            for position in env.positions
        ],
        dtype=np.int64,
    )
    if actions.shape != (len(positions),) or np.any((actions < 0) | (actions >= env.num_actions)):
        raise ValueError("policy returned invalid discrete actions")
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, positions=positions, actions=actions)
    return path


def load_policy_actions(
    snapshot_path: str | Path, positions: tuple[Position, ...], *, num_actions: int = 4
) -> np.ndarray:
    """Read a snapshot and align its actions to ``positions``, never by index assumption."""
    with np.load(snapshot_path, allow_pickle=False) as data:
        saved_positions = data["positions"]
        actions = data["actions"]
    if saved_positions.shape != (len(positions), 2) or saved_positions.dtype.kind not in "iu":
        raise ValueError("policy snapshot positions must be an integer array [states, 2]")
    if actions.shape != (len(positions),) or actions.dtype.kind not in "iu":
        raise ValueError("policy snapshot actions must be an integer vector [states]")
    if np.any((actions < 0) | (actions >= num_actions)):
        raise ValueError("policy snapshot contains invalid actions")
    mapping = {(int(r), int(c)): int(a) for (r, c), a in zip(saved_positions, actions)}
    if len(mapping) != len(positions) or set(mapping) != set(positions):
        raise ValueError("policy snapshot states don't match the environment")
    return np.asarray([mapping[p] for p in positions], dtype=np.int64)


@dataclass(frozen=True, slots=True)
class SnapshotDiagnostics:
    actions: np.ndarray
    diagnostics: PolicyDiagnostics


@dataclass(frozen=True, slots=True)
class TabularPolicyAnalysis:
    """The saved JSON report plus the arrays needed to plot it."""

    report: dict[str, Any]
    oracle: FourRoomsOracle
    snapshots: dict[str, SnapshotDiagnostics]


def analyze_saved_four_rooms_run(run_dir: str | Path, *, gamma: float = 0.99) -> TabularPolicyAnalysis:
    """Compare saved policy snapshots with the exact optimal policy, without torch.

    The original experiment artifacts remain unchanged. The report is derived
    from saved policy snapshots, not from loading or evaluating the model, and
    is written to ``analysis/tabular_policy_v1.json``.
    """
    run = load_four_rooms_run(run_dir)
    oracle = build_four_rooms_oracle(FourRooms(), run.goal, gamma=gamma)
    optimal_policy = np.argmax(oracle.optimal.optimal_actions, axis=1).astype(np.int64)
    optimal_metrics = oracle.compare(optimal_policy, max_steps=run.max_steps)
    report: dict[str, Any] = {
        "schema_version": 1,
        "analysis": "four_rooms_tabular_policy_v1",
        "run_id": run.manifest["run_id"],
        "task_id": run.task_id,
        "goal": list(run.goal),
        "reward_definition": "reward 1 on entering goal; goal terminal; 0 otherwise",
        "pretraining_rewards_used": False,
        "discount": gamma,
        "rollout_horizon": run.max_steps,
        "state_weighting": "uniform over non-goal traversable states",
        "tie_handling": "every action with Q*(s,a) >= max_a Q*(s,a)-1e-8 counts as optimal",
        "optimal_policy": optimal_metrics.summary(),
        "maximum_shortest_path_steps": int(oracle.shortest_distances.max()),
        "snapshots": {},
    }
    snapshots: dict[str, SnapshotDiagnostics] = {}
    for phase in SNAPSHOT_PHASES:
        path = run.root / f"policy_{phase}.npz"
        if not path.exists():
            continue
        actions = load_policy_actions(path, oracle.positions)
        diagnostics = oracle.compare(actions, max_steps=run.max_steps)
        snapshots[phase] = SnapshotDiagnostics(actions, diagnostics)
        report["snapshots"][phase] = {
            "policy_snapshot": path.name,
            **diagnostics.summary(),
            "success_gap_to_optimal": float(
                optimal_metrics.rollout_success_fraction - diagnostics.rollout_success_fraction
            ),
        }
    if not snapshots:
        raise ValueError("no saved policy snapshots were found")
    run.write_report("tabular_policy_v1.json", report)
    return TabularPolicyAnalysis(report, oracle, snapshots)
