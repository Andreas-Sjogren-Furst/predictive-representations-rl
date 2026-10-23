"""Render saved AERIS Four Rooms artifacts without reloading a checkpoint."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from aeris.environments.gridworld import FourRooms


def _plot_lib():
    # Lazy import: core AERIS and training do not require matplotlib.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    return plt, Rectangle


def _draw_map(ax, env: FourRooms, goal: tuple[int, int]):
    _, Rectangle = _plot_lib()
    for r in range(env.HEIGHT):
        for c in range(env.WIDTH):
            ax.add_patch(Rectangle(
                (c - 0.5, r - 0.5), 1, 1,
                facecolor="0.35" if (r, c) not in env.positions else "1.0",
                edgecolor="0.8", linewidth=0.6,
            ))
    gr, gc = goal
    ax.plot(gc, gr, marker="*", markersize=15, label="Goal")
    ax.set_xlim(-0.5, env.WIDTH - 0.5)
    ax.set_ylim(env.HEIGHT - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.set_xticks(range(env.WIDTH))
    ax.set_yticks(range(env.HEIGHT))


def render_policy_map(snapshot_file: str | Path, goal: tuple[int, int],
                      output_file: str | Path) -> Path:
    """Plot deterministic per-state action choices, not optimal action values."""
    plt, _ = _plot_lib()
    env = FourRooms()
    with np.load(snapshot_file, allow_pickle=False) as snap:
        positions = snap["positions"]
        actions = snap["actions"]
    if positions.ndim != 2 or positions.shape[1] != 2 or actions.shape != (len(positions),):
        raise ValueError("invalid policy snapshot shape")
    glyph = {0: "↑", 1: "↓", 2: "←", 3: "→"}
    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    _draw_map(ax, env, goal)
    for (r, c), action in zip(positions, actions):
        pos = (int(r), int(c))
        if pos not in env.positions or int(action) not in glyph:
            raise ValueError("policy snapshot contains invalid state/action")
        if pos != goal:
            ax.text(c, r, glyph[int(action)], ha="center", va="center", fontsize=12)
    ax.set_title(f"Learned policy — {Path(snapshot_file).stem}")
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=145, bbox_inches="tight")
    plt.close(fig)
    return path


def render_trajectory(trajectory: dict, goal: tuple[int, int], output_file: str | Path) -> Path:
    """Show actual recorded action sequence; no simulated reconstruction."""
    plt, _ = _plot_lib()
    env = FourRooms()
    coords = [env.position_of(trajectory["initial_observation"])]
    coords += [env.position_of(s["next_observation"]) for s in trajectory["steps"]]
    rows, cols = zip(*coords)
    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    _draw_map(ax, env, goal)
    ax.plot(cols, rows, "o-", markersize=3, linewidth=1.5, label="Trajectory")
    ax.plot(cols[0], rows[0], marker="s", markersize=8, label="Start")
    ax.set_title(
        f"{trajectory['checkpoint_id']} · episode {trajectory['episode_index']} "
        f"· {'success' if trajectory['success'] else 'not reached'} "
        f"({len(trajectory['steps'])} steps)"
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=145, bbox_inches="tight")
    plt.close(fig)
    return path


def render_training_metrics(metrics_file: str | Path, output_file: str | Path) -> Path:
    plt, _ = _plot_lib()
    records = [json.loads(line) for line in Path(metrics_file).read_text(encoding="utf8").splitlines() if line.strip()]
    if not records:
        raise ValueError("no training metrics to visualize")
    metrics = ("fb_loss", "fb_regularization", "total_loss")
    fig, ax = plt.subplots(figsize=(7, 4))
    plotted = 0
    for name in metrics:
        pairs = [(row["step"], m["value"]) for row in records
                 if row["step_axis"] == "gradient_updates"
                 for m in row["metrics"] if m["name"] == name]
        if pairs:
            ax.plot(*zip(*pairs), label=name)
            plotted += 1
    if not plotted:
        plt.close(fig)
        raise ValueError("no FB metrics found for gradient_updates")
    ax.set_title("FB training diagnostics")
    ax.set_xlabel("Gradient updates")
    ax.set_ylabel("Logged loss")
    ax.grid(alpha=0.25)
    ax.legend()
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=145, bbox_inches="tight")
    plt.close(fig)
    return path


def render_evaluation_success(evaluations_file: str | Path, output_file: str | Path) -> Path:
    """Render recorded evaluation success, keeping checkpoint labels explicit."""
    plt, _ = _plot_lib()
    records = [json.loads(line) for line in Path(evaluations_file).read_text(encoding="utf8").splitlines() if line.strip()]
    entries = [
        (row["checkpoint_id"], float(metric["value"]), row["episode_count"])
        for row in records
        for metric in row["metrics"]
        if metric["name"] == "success_rate"
    ]
    if not entries:
        raise ValueError("no success_rate evaluation metric found")
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(entries))
    ax.bar(x, [e[1] for e in entries])
    ax.set_xticks(x, [f"{ckpt}\n(n={n})" for ckpt, _, n in entries])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Goal success rate")
    ax.set_title("Evaluation (not training performance)")
    for i, (_, value, _) in enumerate(entries):
        ax.text(i, min(value + 0.025, 1.02), f"{value:.1%}", ha="center")
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=145, bbox_inches="tight")
    plt.close(fig)
    return path


def render_four_rooms_report(run_directory: str | Path) -> tuple[Path, ...]:
    """Render recorded artifacts; does not require torch or a live policy."""
    root = Path(run_directory)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf8"))
    if manifest["status"] != "completed":
        raise ValueError("only completed runs can be rendered")
    task = manifest["experiment"]["evaluation"]["targets"][0]["task"]
    goal = (int(task["settings"]["goal_row"]), int(task["settings"]["goal_col"]))
    output = root / "figures"
    output.mkdir(exist_ok=True)
    made: list[Path] = []
    metrics = root / "training_metrics.jsonl"
    if metrics.exists():
        made.append(render_training_metrics(metrics, output / "training.png"))
    evaluations = root / "evaluations.jsonl"
    if evaluations.exists():
        made.append(render_evaluation_success(evaluations, output / "success_rate.png"))
    for phase in ("initial", "final"):
        snapshot = root / f"policy_{phase}.npz"
        if snapshot.exists():
            made.append(render_policy_map(snapshot, goal, output / f"policy_{phase}.png"))
    traces_file = root / "trajectories.jsonl"
    if traces_file.exists():
        traces = [json.loads(line) for line in traces_file.read_text(encoding="utf8").splitlines() if line.strip()]
        for phase, checkpoint in (("initial", "initial_in_memory"), ("final", "checkpoint.pt")):
            first = next((t for t in traces if t["checkpoint_id"] == checkpoint), None)
            if first is not None:
                made.append(render_trajectory(first, goal, output / f"trajectory_{phase}.png"))
    return tuple(made)


def save_policy_snapshot(agent, env: FourRooms, condition, output_file: str | Path) -> Path:
    """Collect deterministic choices from the public Agent.act contract only."""
    from aeris.domain.agent import ActionRequest
    positions = np.asarray(env.positions, dtype=np.int64)
    actions = np.asarray([
        agent.act(ActionRequest(
            observation=env.observation_for(position),
            condition=condition, deterministic=True,
        )).action
        for position in env.positions
    ], dtype=np.int64)
    if actions.shape != (len(positions),) or np.any((actions < 0) | (actions >= env.num_actions)):
        raise ValueError("policy returned invalid discrete actions")
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, positions=positions, actions=actions)
    return path
