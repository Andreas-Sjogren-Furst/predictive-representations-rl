"""Render saved AERIS Four Rooms artifacts without reloading a checkpoint."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from aeris.analysis.four_rooms import SNAPSHOT_PHASES, load_four_rooms_run, load_policy_actions
from aeris.data import read_jsonl
from aeris.environments.gridworld import FourRooms
from aeris.visualization._plotting import ACTION_ARROWS, draw_grid, pyplot, save_figure

# Evaluation checkpoint labels written by examples/fb_four_rooms.py.
TRAJECTORY_CHECKPOINTS = {"initial": "initial_in_memory", "final": "checkpoint.pt"}


def render_policy_map(snapshot_file: str | Path, goal: tuple[int, int], output_file: str | Path) -> Path:
    """Plot deterministic per-state action choices, not optimal action values."""
    env = FourRooms()
    actions = load_policy_actions(snapshot_file, env.positions, num_actions=env.num_actions)
    fig, ax = pyplot().subplots(figsize=(6.4, 6.4))
    draw_grid(ax, env, goal)
    for (r, c), action in zip(env.positions, actions):
        if (r, c) != goal:
            ax.text(c, r, ACTION_ARROWS[int(action)], ha="center", va="center", fontsize=12)
    ax.set_title(f"Learned policy — {Path(snapshot_file).stem}")
    return save_figure(fig, output_file)


def render_trajectory(trajectory: dict[str, Any], goal: tuple[int, int], output_file: str | Path) -> Path:
    """Show the actual recorded action sequence; no simulated reconstruction."""
    env = FourRooms()
    coords = [env.position_of(trajectory["initial_observation"])]
    coords += [env.position_of(s["next_observation"]) for s in trajectory["steps"]]
    rows, cols = zip(*coords)
    fig, ax = pyplot().subplots(figsize=(6.4, 6.4))
    draw_grid(ax, env, goal)
    ax.plot(cols, rows, "o-", markersize=3, linewidth=1.5, label="Trajectory")
    ax.plot(cols[0], rows[0], marker="s", markersize=8, label="Start")
    outcome = "success" if trajectory["success"] else "not reached"
    ax.set_title(
        f"{trajectory['checkpoint_id']} · episode {trajectory['episode_index']} "
        f"· {outcome} ({len(trajectory['steps'])} steps)"
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)
    return save_figure(fig, output_file)


def render_training_metrics(metrics_file: str | Path, output_file: str | Path) -> Path:
    records = read_jsonl(metrics_file)
    if not records:
        raise ValueError("no training metrics to visualize")
    fig, ax = pyplot().subplots(figsize=(7, 4))
    plotted = 0
    for name in ("fb_loss", "fb_regularization", "total_loss"):
        pairs = [
            (row["step"], m["value"])
            for row in records
            if row["step_axis"] == "gradient_updates"
            for m in row["metrics"]
            if m["name"] == name
        ]
        if pairs:
            ax.plot(*zip(*pairs), label=name)
            plotted += 1
    if not plotted:
        pyplot().close(fig)
        raise ValueError("no FB metrics found for gradient_updates")
    ax.set_title("FB training diagnostics")
    ax.set_xlabel("Gradient updates")
    ax.set_ylabel("Logged loss")
    ax.grid(alpha=0.25)
    ax.legend()
    return save_figure(fig, output_file)


def render_evaluation_success(evaluations_file: str | Path, output_file: str | Path) -> Path:
    """Render recorded evaluation success, keeping checkpoint labels explicit."""
    entries = [
        (row["checkpoint_id"], float(metric["value"]), row["episode_count"])
        for row in read_jsonl(evaluations_file)
        for metric in row["metrics"]
        if metric["name"] == "success_rate"
    ]
    if not entries:
        raise ValueError("no success_rate evaluation metric found")
    fig, ax = pyplot().subplots(figsize=(7, 4))
    x = np.arange(len(entries))
    ax.bar(x, [value for _, value, _ in entries])
    ax.set_xticks(x, [f"{checkpoint}\n(n={n})" for checkpoint, _, n in entries])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Goal success rate")
    ax.set_title("Evaluation (not training performance)")
    for i, (_, value, _) in enumerate(entries):
        ax.text(i, min(value + 0.025, 1.02), f"{value:.1%}", ha="center")
    return save_figure(fig, output_file)


def render_four_rooms_report(run_directory: str | Path) -> tuple[Path, ...]:
    """Render every recorded artifact of a run; does not require torch or a live policy."""
    run = load_four_rooms_run(run_directory)
    output = run.root / "figures"
    made: list[Path] = []
    metrics = run.root / "training_metrics.jsonl"
    if metrics.exists():
        made.append(render_training_metrics(metrics, output / "training.png"))
    evaluations = run.root / "evaluations.jsonl"
    if evaluations.exists():
        made.append(render_evaluation_success(evaluations, output / "success_rate.png"))
    for phase in SNAPSHOT_PHASES:
        snapshot = run.root / f"policy_{phase}.npz"
        if snapshot.exists():
            made.append(render_policy_map(snapshot, run.goal, output / f"policy_{phase}.png"))
    traces_file = run.root / "trajectories.jsonl"
    if traces_file.exists():
        traces = read_jsonl(traces_file)
        for phase, checkpoint in TRAJECTORY_CHECKPOINTS.items():
            first = next((t for t in traces if t["checkpoint_id"] == checkpoint), None)
            if first is not None:
                made.append(render_trajectory(first, run.goal, output / f"trajectory_{phase}.png"))
    return tuple(made)
