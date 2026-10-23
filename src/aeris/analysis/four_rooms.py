"""Four Rooms tabular-model adapter and saved-policy diagnostics.

This is deliberately environment-specific. The generic value iteration and
policy-comparison functions in tabular.py have no dependency on FourRooms.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np

from aeris.analysis.tabular import OptimalValues, PolicyDiagnostics, compare_deterministic_policy, value_iteration
from aeris.environments.gridworld import FourRooms


@dataclass(frozen=True, slots=True)
class FourRoomsOracle:
    positions: tuple[tuple[int, int], ...]
    goal: tuple[int, int]
    goal_index: int
    next_state_indices: np.ndarray
    shortest_distances: np.ndarray
    optimal: OptimalValues


def build_four_rooms_oracle(env: FourRooms, goal: tuple[int, int], *, gamma: float = 0.99) -> FourRoomsOracle:
    """Goal reward = 1 on ENTERING the goal; the goal then becomes terminal.

    This diagnostic reward is deliberately separate from the reward-free FB
    pretraining transitions. A reward is NOT repeatedly earned by waiting at
    the goal, nor assumed to have been present during training.
    """
    if goal not in env.positions:
        raise ValueError(f"invalid Four Rooms goal: {goal!r}")
    index = {pos: i for i, pos in enumerate(env.positions)}
    goal_index = index[goal]
    next_states = np.asarray([[index[env.next_position(pos, a)]
                               for a in range(env.num_actions)] for pos in env.positions], dtype=np.int64)
    rewards = (next_states == goal_index).astype(np.float64)
    rewards[goal_index] = 0.0
    done = np.zeros(len(env.positions), dtype=bool)
    done[goal_index] = True
    distances = np.asarray([env.shortest_distance(pos, goal) for pos in env.positions], dtype=np.int64)
    optimal = value_iteration(next_states, rewards, done, gamma=gamma)
    return FourRoomsOracle(env.positions, goal, goal_index, next_states, distances, optimal)


def load_policy_actions(snapshot_path: str | Path, oracle: FourRoomsOracle, *, num_actions: int = 4) -> np.ndarray:
    """Read a snapshot and align its state order to the exact model, never by index assumption."""
    with np.load(snapshot_path, allow_pickle=False) as data:
        positions = data["positions"]
        actions = data["actions"]
    if positions.shape != (len(oracle.positions), 2) or positions.dtype.kind not in "iu":
        raise ValueError("policy snapshot positions must be an integer array [states, 2]")
    if actions.shape != (len(oracle.positions),) or actions.dtype.kind not in "iu":
        raise ValueError("policy snapshot actions must be an integer vector [states]")
    if np.any((actions < 0) | (actions >= num_actions)):
        raise ValueError("policy snapshot contains invalid actions")
    mapping = {(int(r), int(c)): int(a) for (r, c), a in zip(positions, actions)}
    if len(mapping) != len(oracle.positions) or set(mapping) != set(oracle.positions):
        raise ValueError("policy snapshot states don't match the environment")
    return np.asarray([mapping[p] for p in oracle.positions], dtype=np.int64)


def compare_four_rooms_snapshot(
    snapshot_path: str | Path,
    oracle: FourRoomsOracle,
    *,
    max_steps: int = 60,
) -> tuple[np.ndarray, PolicyDiagnostics]:
    actions = load_policy_actions(snapshot_path, oracle)
    diagnostics = compare_deterministic_policy(
        actions, oracle.next_state_indices, oracle.optimal,
        goal_index=oracle.goal_index, max_steps=max_steps,
        shortest_distances=oracle.shortest_distances,
    )
    return actions, diagnostics


def analyze_saved_four_rooms_run(run_dir: str | Path, *, gamma: float = 0.99) -> dict:
    """Compute and store numerical results from a completed run, no graphics or torch.

    The original experiment artifacts remain unchanged. This report is derived
    from saved policy snapshots, not from evaluating/loading the learned model.
    """
    import json

    root = Path(run_dir)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "completed":
        raise ValueError("run must be completed before analysis")
    exp = manifest["experiment"]
    if exp["environment"]["name"] != "four_rooms":
        raise ValueError("this analysis only supports the Four Rooms environment")
    targets = exp["evaluation"]["targets"]
    if len(targets) != 1:
        raise ValueError("this diagnostic currently requires exactly one evaluation target")
    settings = targets[0]["task"]["settings"]
    goal = (int(settings["goal_row"]), int(settings["goal_col"]))
    horizon = int(exp["evaluation"]["protocol"]["settings"]["max_steps"])
    env = FourRooms()
    oracle = build_four_rooms_oracle(env, goal, gamma=gamma)
    optimal_policy = np.argmax(oracle.optimal.optimal_actions, axis=1).astype(np.int64)
    optimal_metrics = compare_deterministic_policy(
        optimal_policy, oracle.next_state_indices, oracle.optimal,
        goal_index=oracle.goal_index, max_steps=horizon,
        shortest_distances=oracle.shortest_distances,
    )
    report = {
        "schema_version": 1,
        "analysis": "four_rooms_tabular_policy_v1",
        "run_id": manifest["run_id"],
        "task_id": targets[0]["task"]["task_id"],
        "goal": list(goal),
        "reward_definition": "reward 1 on entering goal; goal terminal; 0 otherwise",
        "pretraining_rewards_used": False,
        "discount": gamma,
        "rollout_horizon": horizon,
        "state_weighting": "uniform over non-goal traversable states",
        "tie_handling": "every action with Q*(s,a) >= max_a Q*(s,a)-1e-8 counts as optimal",
        "optimal_policy": optimal_metrics.summary(),
        "maximum_shortest_path_steps": int(oracle.shortest_distances.max()),
        "snapshots": {},
    }
    for phase in ("initial", "final"):
        snapshot = root / f"policy_{phase}.npz"
        if not snapshot.exists():
            continue
        _, diagnostics = compare_four_rooms_snapshot(snapshot, oracle, max_steps=horizon)
        report["snapshots"][phase] = {
            "policy_snapshot": snapshot.name,
            **diagnostics.summary(),
            "success_gap_to_optimal": float(optimal_metrics.rollout_success_fraction - diagnostics.rollout_success_fraction),
        }
    if not report["snapshots"]:
        raise ValueError("no saved policy snapshots were found")
    out = root / "analysis" / "tabular_policy_v1.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report
