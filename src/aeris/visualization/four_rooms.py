"""Render saved AERIS Four Rooms artifacts without reloading a checkpoint."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np

from aeris.analysis.four_rooms import SNAPSHOT_PHASES, load_four_rooms_run
from aeris.domain.artifacts import PolicySnapshot
from aeris.domain.results import EvaluationRecord, TrainingMetricRecord
from aeris.domain.trajectory import EvaluationTrajectory
from aeris.environments.gridworld import FourRooms
from aeris.visualization._plotting import ACTION_ARROWS, draw_grid, pyplot, save_figure

# Evaluation checkpoint labels written by examples/fb_four_rooms.py.
TRAJECTORY_CHECKPOINTS = {"initial": "initial_in_memory", "final": "checkpoint.pt"}


def render_policy_map(
    snapshot: PolicySnapshot,
    goal: tuple[int, int],
    output_file: str | Path,
    *,
    name: str = "policy",
) -> Path:
    """Plot deterministic per-state action choices, not optimal action values."""
    env = FourRooms()
    actions = snapshot.actions_for(env.positions)
    fig, ax = pyplot().subplots(figsize=(6.4, 6.4))
    draw_grid(ax, env, goal)
    for (r, c), action in zip(env.positions, actions, strict=True):
        if (r, c) != goal:
            ax.text(c, r, ACTION_ARROWS[int(action)], ha="center", va="center", fontsize=12)
    ax.set_title(f"Learned policy — {name}")
    return save_figure(fig, output_file)


def render_trajectory(
    trajectory: EvaluationTrajectory, goal: tuple[int, int], output_file: str | Path
) -> Path:
    """Show the actual recorded action sequence; no simulated reconstruction."""
    env = FourRooms()
    coords = [env.position_of(trajectory.initial_observation)]
    coords += [env.position_of(step.next_observation) for step in trajectory.steps]
    rows, cols = zip(*coords, strict=True)
    fig, ax = pyplot().subplots(figsize=(6.4, 6.4))
    draw_grid(ax, env, goal)
    ax.plot(cols, rows, "o-", markersize=3, linewidth=1.5, label="Trajectory")
    ax.plot(cols[0], rows[0], marker="s", markersize=8, label="Start")
    outcome = "success" if trajectory.success else "not reached"
    ax.set_title(
        f"{trajectory.checkpoint_id} · episode {trajectory.episode_index} "
        f"· {outcome} ({trajectory.length} steps)"
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)
    return save_figure(fig, output_file)


def render_training_metrics(
    records: Iterable[TrainingMetricRecord], output_file: str | Path
) -> Path:
    records = list(records)
    if not records:
        raise ValueError("no training metrics to visualize")
    fig, ax = pyplot().subplots(figsize=(7, 4))
    plotted = 0
    for name in ("fb_loss", "fb_regularization", "total_loss"):
        pairs = [
            (record.step, metric.value)
            for record in records
            if record.step_axis == "gradient_updates"
            for metric in record.metrics
            if metric.name == name
        ]
        if pairs:
            ax.plot(*zip(*pairs, strict=True), label=name)
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


def render_evaluation_success(records: Iterable[EvaluationRecord], output_file: str | Path) -> Path:
    """Render recorded evaluation success, keeping checkpoint labels explicit."""
    entries = [
        (record.checkpoint_id, metric.value, record.episode_count)
        for record in records
        for metric in record.metrics
        if metric.name == "success_rate"
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
    store, output = run.store, run.store.figures_dir
    made: list[Path] = []
    training = list(store.read_training_metrics())
    if training:
        made.append(render_training_metrics(training, output / "training.png"))
    evaluations = list(store.read_evaluations())
    if evaluations:
        made.append(render_evaluation_success(evaluations, output / "success_rate.png"))
    for phase in SNAPSHOT_PHASES:
        snapshot = store.read_policy_snapshot(phase)
        if snapshot is not None:
            name = store.policy_snapshot_path(phase).stem
            made.append(render_policy_map(snapshot, run.goal, output / f"{name}.png", name=name))
    trajectories = list(store.read_trajectories())
    for phase, checkpoint in TRAJECTORY_CHECKPOINTS.items():
        first = next((t for t in trajectories if t.checkpoint_id == checkpoint), None)
        if first is not None:
            made.append(render_trajectory(first, run.goal, output / f"trajectory_{phase}.png"))
    return tuple(made)
