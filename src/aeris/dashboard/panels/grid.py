"""Plotly drawing of the Four Rooms grid, shared by the map panels."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, TypeAlias

import numpy as np
import plotly.graph_objects as go

from aeris.dashboard.data import RunData
from aeris.environments.four_rooms import FourRooms, Position

ColourScale: TypeAlias = str | Sequence[Sequence[Any]]  # a Plotly name or [[stop, colour], ...]
ARROWS = {0: "↑", 1: "↓", 2: "←", 3: "→"}
ENV = FourRooms()

# Map colours. Backgrounds under arrows stay light so ink arrows remain legible.
INK, MUTED = "#0b0b0b", "#52514e"
REWARD_SCALE = [[0.0, "#fcfcfb"], [1.0, "#86b6ef"]]  # sequential, light: low -> high reward
REWARD_DIVERGING = [[0.0, "#ef9f9a"], [0.5, "#f0efec"], [1.0, "#86b6ef"]]  # negative < 0 < positive
COUNT_SCALE = [[0.0, "#cde2fb"], [1.0, "#104281"]]  # sequential blue
SCORE_SCALE = [[0.0, "#b02b2b"], [0.5, "#f0efec"], [1.0, "#1c5cab"]]  # worse than random < 0 < 1


def reward_colours(rewards: Sequence[float] | np.ndarray) -> dict[str, Any]:
    """Heatmap range and scale for a reward map: diverging around 0 when rewards take both
    signs, sequential from the lowest to the highest reward otherwise."""
    low, high = float(np.min(rewards)), float(np.max(rewards))
    if low < 0 < high:
        bound = max(-low, high)
        return {"zmin": -bound, "zmax": bound, "colorscale": REWARD_DIVERGING}
    return {"zmin": low if low < high else low - 1, "zmax": high, "colorscale": REWARD_SCALE}


def cell_label(state: tuple[int, ...]) -> str:
    """A cell as an option value: ``"row,col"``."""
    return f"{state[0]},{state[1]}"


def parse_cell(cell: str) -> tuple[int, int]:
    row, col = cell.split(",")
    return int(row), int(col)


def is_four_rooms(run: RunData) -> bool:
    environment = run.manifest.experiment.environment
    return environment is not None and environment.name == "four_rooms"


def run_goal(run: RunData) -> Position | None:
    evaluation = run.manifest.experiment.evaluation
    if evaluation is None or len(evaluation.targets) != 1:
        return None
    return ENV.goal_from_task(evaluation.targets[0].task)


def add_grid(
    fig: go.Figure,
    values: Mapping[tuple[int, ...], float] | None = None,
    *,
    row: int | None = None,
    col: int | None = None,
    zmin: float | None = 0.0,
    zmax: float | None = 1.0,
    colorscale: ColourScale = "Viridis",
    colorbar: dict | None = None,
    text_format: str | None = None,
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
                **({"texttemplate": text_format, "textfont": {"size": 10}} if text_format else {}),
            ),
            **place,
        )
    axis = {"showgrid": False, "zeroline": False, "dtick": 1}
    fig.update_xaxes(range=[-0.5, ENV.WIDTH - 0.5], **axis, **place)
    fig.update_yaxes(range=[ENV.HEIGHT - 0.5, -0.5], **axis, **place)


def lock_aspect(fig: go.Figure, subplots: int = 1) -> None:
    """Square grid cells in every subplot; the plot area shrinks to fit rather than the axes
    stretching past the grid."""
    for index in range(1, subplots + 1):
        suffix = "" if index == 1 else str(index)
        fig.update_layout(
            {
                f"xaxis{suffix}": {"constrain": "domain"},
                f"yaxis{suffix}": {
                    "scaleanchor": f"x{suffix}",
                    "scaleratio": 1,
                    "constrain": "domain",
                },
            }
        )
    fig.update_layout(plot_bgcolor="white")


def grid_figure(
    values: Mapping[tuple[int, ...], float] | None = None,
    *,
    zmin: float = 0.0,
    zmax: float = 1.0,
    colorscale: ColourScale = "Viridis",
    colorbar_title: str = "",
    text_format: str | None = None,
) -> go.Figure:
    """One grid: walls in grey, optional per-cell values as a heatmap, rows growing downwards.

    ``text_format`` (a Plotly texttemplate such as ``"%{z:.2f}"``) prints each value in its cell.
    """
    fig = go.Figure()
    add_grid(
        fig,
        values,
        zmin=zmin,
        zmax=zmax,
        colorscale=colorscale,
        colorbar={"title": colorbar_title, "thickness": 12},
        text_format=text_format,
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


def add_playback(fig: go.Figure, frames: list[go.Frame], *, duration: int = 150) -> None:
    """Animate ``fig`` through ``frames`` (named by step) with play/pause buttons and a slider."""
    fig.frames = frames
    fig.update_layout(
        updatemenus=[
            {
                "type": "buttons",
                "x": 0,
                "y": -0.08,
                "xanchor": "left",
                "direction": "left",
                "buttons": [
                    {
                        "label": "▶ play",
                        "method": "animate",
                        "args": [
                            None,
                            {"frame": {"duration": duration, "redraw": False}, "fromcurrent": True},
                        ],
                    },
                    {
                        "label": "❚❚",
                        "method": "animate",
                        "args": [[None], {"frame": {"duration": 0}, "mode": "immediate"}],
                    },
                ],
            }
        ],
        sliders=[
            {
                "steps": [
                    {
                        "label": frame.name,
                        "method": "animate",
                        "args": [
                            [frame.name],
                            {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"},
                        ],
                    }
                    for frame in frames
                ],
                "x": 0.18,
                "len": 0.82,
                "y": -0.04,
                "currentvalue": {"prefix": "step ", "xanchor": "right"},  # clear of the buttons
            }
        ],
    )
