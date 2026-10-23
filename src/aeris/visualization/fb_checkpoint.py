"""Optional, reproducible plots from saved numerical FB diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from aeris.environments import FourRooms
from aeris.visualization._plotting import (
    GOAL_COLOR,
    WALL_COLOR,
    pyplot,
    save_figure,
    style_grid_axes,
)


def render_fb_action_alignment(report: dict[str, Any], output_file: str | Path) -> Path:
    """Map whether FB's greedy choice improves its own target-Boltzmann policy.

    This is NOT a map of globally optimal actions for terminal navigation.
    The report is sufficient; no neural model or training environment is loaded.
    """
    from matplotlib.colors import ListedColormap

    env = FourRooms()
    positions = report["state_positions"]
    agreement = report["state_action_improvement_agreement"]
    if len(positions) != len(env.positions) or len(agreement) != len(positions):
        raise ValueError("malformed statewise FB diagnostic")
    grid = np.full((env.HEIGHT, env.WIDTH), np.nan, dtype=np.float64)
    expected = set(env.positions)
    visited = set()
    for pos, good in zip(positions, agreement, strict=True):
        point = (int(pos[0]), int(pos[1]))
        if point not in expected or point in visited or type(good) is not bool:
            raise ValueError("invalid or duplicate statewise FB diagnostic")
        visited.add(point)
        grid[point] = float(good)
    if visited != expected:
        raise ValueError("statewise FB diagnostic does not cover the maze")

    goal = tuple(report["goal"])
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
