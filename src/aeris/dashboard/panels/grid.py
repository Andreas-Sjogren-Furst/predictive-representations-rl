"""Plotly drawing of the Four Rooms grid, shared by the map panels."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import plotly.graph_objects as go

from aeris.dashboard.data import RunData
from aeris.environments.four_rooms import FourRooms, Position

ARROWS = {0: "↑", 1: "↓", 2: "←", 3: "→"}
ENV = FourRooms()


def is_four_rooms(run: RunData) -> bool:
    environment = run.manifest.experiment.environment
    return environment is not None and environment.name == "four_rooms"


def run_goal(run: RunData) -> Position | None:
    evaluation = run.manifest.experiment.evaluation
    if evaluation is None or len(evaluation.targets) != 1:
        return None
    return ENV.goal_from_task(evaluation.targets[0].task)


def grid_figure(
    values: Mapping[tuple[int, ...], float] | None = None,
    *,
    zmin: float = 0.0,
    zmax: float = 1.0,
    colorscale: str = "Viridis",
    colorbar_title: str = "",
) -> go.Figure:
    """Walls in grey, optional per-cell values as a heatmap, rows growing downwards."""
    free = set(ENV.positions)
    walls = np.array(
        [[np.nan if (r, c) in free else 1.0 for c in range(ENV.WIDTH)] for r in range(ENV.HEIGHT)]
    )
    fig = go.Figure(
        go.Heatmap(
            z=walls, colorscale=[[0, "#5a5a5a"], [1, "#5a5a5a"]], showscale=False, hoverinfo="skip"
        )
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
                colorbar={"title": colorbar_title, "thickness": 12},
                hovertemplate="row %{y}, col %{x}: %{z:.2f}<extra></extra>",
            )
        )
    fig.update_xaxes(range=[-0.5, ENV.WIDTH - 0.5], dtick=1, showgrid=False, zeroline=False)
    fig.update_yaxes(
        range=[ENV.HEIGHT - 0.5, -0.5],
        dtick=1,
        showgrid=False,
        zeroline=False,
        scaleanchor="x",
        scaleratio=1,
    )
    fig.update_layout(plot_bgcolor="white")
    return fig


def add_goal(fig: go.Figure, goal: Position) -> None:
    fig.add_trace(
        go.Scatter(
            x=[goal[1]],
            y=[goal[0]],
            mode="markers",
            name="goal",
            marker={"symbol": "star", "size": 18, "color": "#e4a72e", "line": {"width": 1}},
        )
    )
