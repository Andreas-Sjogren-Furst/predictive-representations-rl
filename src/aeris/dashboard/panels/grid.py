"""Plotly drawing of the Four Rooms grid, shared by the map panels."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import plotly.graph_objects as go

from aeris.dashboard.data import RunData
from aeris.environments.four_rooms import FourRooms, Position

ARROWS = {0: "↑", 1: "↓", 2: "←", 3: "→"}
ENV = FourRooms()
DOORWAYS = {(5, 2): "doorway", (5, 8): "doorway", (2, 5): "doorway", (8, 5): "doorway"}


def is_four_rooms(run: RunData) -> bool:
    environment = run.manifest.experiment.environment
    return environment is not None and environment.name == "four_rooms"


def run_goal(run: RunData) -> Position | None:
    evaluation = run.manifest.experiment.evaluation
    if evaluation is None or len(evaluation.targets) != 1:
        return None
    return ENV.goal_from_task(evaluation.targets[0].task)


def room_of(cell: tuple[int, ...]) -> str:
    if tuple(cell) in DOORWAYS:
        return "doorway"
    row, col = cell
    return f"{'top' if row < 5 else 'bottom'}-{'left' if col < 5 else 'right'}"


def add_grid(
    fig: go.Figure,
    values: Mapping[tuple[int, ...], float] | None = None,
    *,
    row: int | None = None,
    col: int | None = None,
    zmin: float | None = 0.0,
    zmax: float | None = 1.0,
    colorscale: str = "Viridis",
    colorbar: dict | None = None,
) -> None:
    """Draw walls in grey and optional per-cell values into ``fig`` (or one of its subplots)."""
    free = set(ENV.positions)
    walls = np.array(
        [[np.nan if (r, c) in free else 1.0 for c in range(ENV.WIDTH)] for r in range(ENV.HEIGHT)]
    )
    place = {"row": row, "col": col} if row is not None else {}
    fig.add_trace(
        go.Heatmap(
            z=walls, colorscale=[[0, "#5a5a5a"], [1, "#5a5a5a"]], showscale=False, hoverinfo="skip"
        ),
        **place,
    )
    if values is not None:
        z = np.full((ENV.HEIGHT, ENV.WIDTH), np.nan)
        for (r, c), value in values.items():
            z[r, c] = value
        fig.add_trace(
            go.Heatmap(
                z=z,
                zmin=zmin,
                zmax=zmax,
                colorscale=colorscale,
                colorbar=colorbar or {"thickness": 12},
                hovertemplate="row %{y}, col %{x}: %{z:.3g}<extra></extra>",
            ),
            **place,
        )
    axis = {"showgrid": False, "zeroline": False, "dtick": 1}
    fig.update_xaxes(range=[-0.5, ENV.WIDTH - 0.5], **axis, **place)
    fig.update_yaxes(range=[ENV.HEIGHT - 0.5, -0.5], **axis, **place)


def lock_aspect(fig: go.Figure, subplots: int = 1) -> None:
    """Square grid cells in every subplot."""
    for index in range(1, subplots + 1):
        suffix = "" if index == 1 else str(index)
        fig.update_layout({f"yaxis{suffix}": {"scaleanchor": f"x{suffix}", "scaleratio": 1}})
    fig.update_layout(plot_bgcolor="white")


def grid_figure(
    values: Mapping[tuple[int, ...], float] | None = None,
    *,
    zmin: float = 0.0,
    zmax: float = 1.0,
    colorscale: str = "Viridis",
    colorbar_title: str = "",
) -> go.Figure:
    """One grid: walls in grey, optional per-cell values as a heatmap, rows growing downwards."""
    fig = go.Figure()
    add_grid(
        fig,
        values,
        zmin=zmin,
        zmax=zmax,
        colorscale=colorscale,
        colorbar={"title": colorbar_title, "thickness": 12},
    )
    lock_aspect(fig)
    return fig


def add_goal(
    fig: go.Figure, goal: tuple[int, ...], *, row: int | None = None, col: int | None = None
) -> None:
    place = {"row": row, "col": col} if row is not None else {}
    fig.add_trace(
        go.Scatter(
            x=[goal[1]],
            y=[goal[0]],
            mode="markers",
            name="goal",
            showlegend=row in (None, 1) and col in (None, 1),
            marker={"symbol": "star", "size": 16, "color": "#e4a72e", "line": {"width": 1}},
        ),
        **place,
    )
