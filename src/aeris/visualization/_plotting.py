"""Shared matplotlib helpers for Four Rooms figures (imported lazily)."""

from __future__ import annotations

from pathlib import Path

from aeris.environments.gridworld import FourRooms

ACTION_ARROWS = {0: "↑", 1: "↓", 2: "←", 3: "→"}
WALL_COLOR = "0.35"
GOAL_COLOR = "#e4a72e"


def pyplot():
    """Import pyplot with a headless backend; core AERIS does not require matplotlib."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def style_grid_axes(ax, env: FourRooms) -> None:
    ax.set_xlim(-0.5, env.WIDTH - 0.5)
    ax.set_ylim(env.HEIGHT - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.set_xticks(range(env.WIDTH))
    ax.set_yticks(range(env.HEIGHT))


def draw_grid(ax, env: FourRooms, goal: tuple[int, int]) -> None:
    """Draw walls, free cells and the goal marker."""
    from matplotlib.patches import Rectangle

    free = set(env.positions)
    for r in range(env.HEIGHT):
        for c in range(env.WIDTH):
            ax.add_patch(
                Rectangle(
                    (c - 0.5, r - 0.5), 1, 1,
                    facecolor="1.0" if (r, c) in free else WALL_COLOR,
                    edgecolor="0.8", linewidth=0.6,
                )
            )
    ax.plot(goal[1], goal[0], marker="*", markersize=15, label="Goal")
    style_grid_axes(ax, env)


def save_figure(fig, output_file: str | Path, *, dpi: int = 145) -> Path:
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    pyplot().close(fig)
    return path
