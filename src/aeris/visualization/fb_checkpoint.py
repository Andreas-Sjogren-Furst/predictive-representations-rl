"""Optional, reproducible plots from saved numerical FB diagnostics."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from aeris.agents.fb.reports import FBCheckpointReport
from aeris.environments import FourRooms
from aeris.visualization._plotting import (
    GOAL_COLOR,
    WALL_COLOR,
    pyplot,
    save_figure,
    style_grid_axes,
)


def render_fb_action_alignment(report: FBCheckpointReport, output_file: str | Path) -> Path:
    """Map whether FB's greedy choice improves its own target-Boltzmann policy.

    This is NOT a map of globally optimal actions for terminal navigation.
    The report is sufficient; no neural model or training environment is loaded.
    """
    from matplotlib.colors import ListedColormap

    env = FourRooms()
    if len(report.state_positions) != len(env.positions) or set(report.state_positions) != set(
        env.positions
    ):
        raise ValueError("statewise FB diagnostic does not cover the maze")
    grid = np.full((env.HEIGHT, env.WIDTH), np.nan, dtype=np.float64)
    for point, good in zip(
        report.state_positions, report.state_action_improvement_agreement, strict=True
    ):
        grid[point] = float(good)
    expected = set(env.positions)

    goal = report.goal
    fig, ax = pyplot().subplots(figsize=(6.2, 6.2))
    ax.set_facecolor(WALL_COLOR)
    ax.imshow(
        np.ma.masked_invalid(grid),
        cmap=ListedColormap(["#dc7777", "#6eb79a"]),
        vmin=0,
        vmax=1,
        interpolation="nearest",
        origin="upper",
    )
    for r, c in expected:
        if (r, c) != goal:
            ax.text(c, r, "✓" if bool(grid[r, c]) else "×", ha="center", va="center", fontsize=10)
    ax.plot(goal[1], goal[0], marker="*", markersize=16, color=GOAL_COLOR)
    ax.grid(color="0.85", lw=0.5)
    style_grid_axes(ax, env)
    ax.set_title("FB action agreement with exact target-policy improvement")
    return save_figure(fig, output_file)
