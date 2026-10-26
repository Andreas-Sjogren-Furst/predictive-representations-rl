"""FB representation panels, drawn from a saved ``fb_representations_v1`` artifact.

Create the artifact with ``aeris analysis representations RUN``; these panels
never load a checkpoint.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from aeris.agents.fb.representations import FBRepresentations
from aeris.analysis.goal_policy import build_goal_oracle
from aeris.analysis.tabular_occupancy import centered_pearson
from aeris.dashboard.data import RunData
from aeris.dashboard.panels.base import STYLE, Choice, Option, Output, Panel, Toggle
from aeris.dashboard.panels.grid import (
    ARROWS,
    ENV,
    add_goal,
    add_grid,
    cell_label,
    lock_aspect,
    parse_cell,
    run_goal,
)
from aeris.environments.four_rooms import room_of

POLICY = "policy (target Boltzmann)"
FORWARD = "F(s, π, z): forward, policy-averaged"
PROFILE = "F·B: learned occupancy profile"
BACKWARD = "B(s): backward"
REPRESENTATIONS = (FORWARD, PROFILE, BACKWARD)
ROOM_ORDER = ("top-left", "top-right", "bottom-left", "bottom-right", "doorway")
COLOURINGS = ("room", "distance to goal", "row", "column", "state density")


def _reps(run: RunData) -> FBRepresentations:
    assert run.representations is not None
    return run.representations


def _goal_choice(run: RunData) -> Choice:
    goals = tuple(cell_label(g) for g in _reps(run).goals)
    goal = run_goal(run)
    default = cell_label(goal) if goal is not None and cell_label(goal) in goals else goals[0]
    return Choice("goal", "Goal (z = B(goal))", goals, default)


def _features(reps: FBRepresentations, kind: str, goal: int) -> np.ndarray:
    """One row per state of the chosen representation, shape [S, k].

    B is near-orthonormal by construction (the regularizer drives E[B Bᵀ] to I),
    so its geometry is nearly structureless; F and F·B carry the dynamics.
    """
    policy = reps.target_policy[goal].astype(np.float64)  # [S, A]
    if kind == BACKWARD:
        return reps.backward.astype(np.float64)
    if kind == FORWARD:
        return np.einsum("sa,sda->sd", policy, reps.forward[goal].astype(np.float64))
    return np.einsum("sa,sat->st", policy, reps.predicted_density(goal).astype(np.float64))


def _correlation(x: np.ndarray, y: np.ndarray) -> str:
    keep = np.isfinite(x) & np.isfinite(y)
    value = centered_pearson(x[keep], y[keep]) if keep.sum() > 1 else None
    return "n/a" if value is None else f"{value:.2f}"


def _short(kind: str) -> str:
    return kind.split(":")[0]


@dataclass(frozen=True)
class RepresentationPanel(Panel):
    """Shared requirement: a saved representation artifact."""

    def missing(self, runs: Sequence[RunData]) -> str | None:
        if runs[0].representations is None:
            return f"No saved representations. Run: aeris analysis representations {runs[0].root}"
        return None


@dataclass(frozen=True)
class RepresentationProjection(RepresentationPanel):
    name = "projection"
    title = "Representation projection"
    scope = "run"
    description = "PCA of every state's representation, embedding-projector style; z = B(goal)."

    def options(self, runs: Sequence[RunData]) -> tuple[Option, ...]:
        return (
            Choice("representation", "Representation", REPRESENTATIONS, FORWARD),
            Choice("dimensions", "Dimensions", (2, 3), 2),
            Choice("color", "Colour by", COLOURINGS, "room"),
            _goal_choice(runs[0]),
        )

    def render(self, runs: Sequence[RunData], options: Mapping[str, Any]) -> Output:
        reps = _reps(runs[0])
        goal = reps.goal_index(parse_cell(options["goal"]))
        features = _features(reps, options["representation"], goal)
        centered = features - features.mean(axis=0)
        _, singular, vectors = np.linalg.svd(centered, full_matrices=False)
        dims = int(options["dimensions"])
        coords = centered @ vectors[:dims].T
        explained = singular**2 / (singular**2).sum()
        labels = [f"({r},{c}) {room_of((r, c))}" for r, c in reps.states]
        fig = go.Figure()
        colour = self._colour(reps, options)
        if options["color"] == "room":
            for room in ROOM_ORDER:
                mask = np.array([room_of(s) == room for s in reps.states])
                if mask.any():
                    fig.add_trace(self._scatter(coords[mask], np.array(labels)[mask], room, None))
        else:
            fig.add_trace(self._scatter(coords, np.array(labels), options["color"], colour))
        if dims == 2:
            fig.update_xaxes(title="PC1")
            fig.update_yaxes(title="PC2")
        share = ", ".join(f"PC{i + 1} {explained[i]:.0%}" for i in range(dims))
        return STYLE.apply(
            fig, f"{_short(options['representation'])} projection · variance {share}"
        )

    @staticmethod
    def _colour(reps: FBRepresentations, options: Mapping[str, Any]) -> np.ndarray | None:
        kind = options["color"]
        if kind == "room":
            return None
        if kind == "row":
            return np.array([s[0] for s in reps.states])
        if kind == "column":
            return np.array([s[1] for s in reps.states])
        if kind == "state density":
            return reps.state_density
        dynamics = ENV.tabular_dynamics()
        distance = dynamics.shortest_path_lengths(dynamics.index_of(parse_cell(options["goal"])))
        return np.array([distance[dynamics.index_of(s)] for s in reps.states])

    @staticmethod
    def _scatter(
        coords: np.ndarray, text: np.ndarray, name: str, colour: np.ndarray | None
    ) -> go.Scatter | go.Scatter3d:
        marker: dict[str, Any] = {"size": 7 if coords.shape[1] == 2 else 4}
        if colour is not None:
            marker.update(
                color=colour, colorscale="Viridis", showscale=True, colorbar={"thickness": 12}
            )
        common = {
            "mode": "markers",
            "marker": marker,
            "name": name,
            "text": list(text),
            "hovertemplate": "%{text}<extra></extra>",
        }
        if coords.shape[1] == 3:
            return go.Scatter3d(x=coords[:, 0], y=coords[:, 1], z=coords[:, 2], **common)
        return go.Scatter(x=coords[:, 0], y=coords[:, 1], **common)


@dataclass(frozen=True)
class RepresentationSimilarity(RepresentationPanel):
    name = "similarity"
    title = "Representation similarity"
    scope = "run"
    description = "Similarity between every pair of states' representations, ordered room by room."

    def options(self, runs: Sequence[RunData]) -> tuple[Option, ...]:
        return (
            Choice("representation", "Representation", REPRESENTATIONS, FORWARD),
            Choice("measure", "Measure", ("cosine", "dot product"), "cosine"),
            _goal_choice(runs[0]),
        )

    def render(self, runs: Sequence[RunData], options: Mapping[str, Any]) -> Output:
        reps = _reps(runs[0])
        order = sorted(
            range(len(reps.states)),
            key=lambda i: (ROOM_ORDER.index(room_of(reps.states[i])), reps.states[i]),
        )
        features = _features(
            reps, options["representation"], reps.goal_index(parse_cell(options["goal"]))
        )
        rows = features[order]
        if options["measure"] == "cosine":
            rows = rows / np.maximum(np.linalg.norm(rows, axis=1, keepdims=True), 1e-12)
        similarity = rows @ rows.T
        ticks = [f"{cell_label(reps.states[i])} {room_of(reps.states[i])}" for i in order]
        limit = float(np.abs(similarity).max())
        fig = go.Figure(
            go.Heatmap(
                z=similarity,
                x=ticks,
                y=ticks,
                colorscale="RdBu",
                zmid=0,
                zmin=-limit,
                zmax=limit,
                hovertemplate="%{y} vs %{x}: %{z:.2f}<extra></extra>",
                colorbar={"thickness": 12},
            )
        )
        fig.update_yaxes(autorange="reversed", showticklabels=False)
        fig.update_xaxes(showticklabels=False)
        title = (
            f"{_short(options['representation'])} {options['measure']} similarity · ordered by room"
        )
        return STYLE.apply(fig, title)


@dataclass(frozen=True)
class BSpectrum(RepresentationPanel):
    name = "b_spectrum"
    title = "B spectrum"
    scope = "run"
    description = "Singular values of centred B: how many directions the representation uses."

    def options(self, runs: Sequence[RunData]) -> tuple[Option, ...]:
        return (Toggle("log", "Log scale", True),)

    def render(self, runs: Sequence[RunData], options: Mapping[str, Any]) -> Output:
        reps = _reps(runs[0])
        singular = np.linalg.svd(reps.backward - reps.backward.mean(axis=0), compute_uv=False)
        energy = singular**2 / (singular**2).sum()
        positive = energy[energy > 0]
        effective_rank = float(np.exp(-(positive * np.log(positive)).sum()))
        fig = go.Figure(
            go.Bar(x=np.arange(1, len(singular) + 1), y=singular, name="singular value")
        )
        fig.update_xaxes(title="component")
        fig.update_yaxes(title="singular value", type="log" if options["log"] else "linear")
        d = reps.backward.shape[1]
        return STYLE.apply(fig, f"{self.title} · effective rank {effective_rank:.1f} of d = {d}")


@dataclass(frozen=True)
class OccupancyMaps(RepresentationPanel):
    name = "occupancy_maps"
    title = "Occupancy: learned vs exact"
    scope = "run"
    description = "Learned F(s0, a, z)·B(t) next to the exact occupancy density m/ρ from one start."

    def options(self, runs: Sequence[RunData]) -> tuple[Option, ...]:
        starts = tuple(cell_label(s) for s in _reps(runs[0]).states)
        return (
            _goal_choice(runs[0]),
            Choice("start", "Start cell s0", starts, starts[0]),
            Choice("action", "First action", (POLICY, *ARROWS.values()), POLICY),
            Toggle("log", "Log scale", False),
        )

    def render(self, runs: Sequence[RunData], options: Mapping[str, Any]) -> Output:
        reps = _reps(runs[0])
        goal = reps.goal_index(parse_cell(options["goal"]))
        start = reps.state_index(parse_cell(options["start"]))
        predicted, exact = (
            reps.predicted_density(goal)[start],
            reps.exact_density[goal][start],
        )  # [A, S]
        if options["action"] == POLICY:
            weights = reps.target_policy[goal][start]
            learned, reference = weights @ predicted, weights @ exact
        else:
            action = list(ARROWS.values()).index(options["action"])
            learned, reference = predicted[action], exact[action]
        if options["log"]:
            learned = np.log10(np.clip(learned, 1e-3, None))
            reference = np.log10(np.clip(reference, 1e-3, None))
        finite = np.concatenate([learned[np.isfinite(learned)], reference[np.isfinite(reference)]])
        low, high = float(finite.min()), float(finite.max())
        fig = make_subplots(
            rows=1, cols=2, subplot_titles=("learned F·B", "exact m/ρ"), horizontal_spacing=0.08
        )
        for col, values in enumerate((learned, reference), start=1):
            cells = {state: float(v) for state, v in zip(reps.states, values, strict=True)}
            colorbar = {"thickness": 12, "title": "log10" if options["log"] else ""}
            add_grid(fig, cells, row=1, col=col, zmin=low, zmax=high, colorbar=colorbar)
            add_goal(fig, reps.goals[goal], row=1, col=col)
            fig.add_trace(
                go.Scatter(
                    x=[reps.states[start][1]],
                    y=[reps.states[start][0]],
                    mode="markers",
                    marker={
                        "symbol": "circle-open",
                        "size": 16,
                        "color": "white",
                        "line": {"width": 3},
                    },
                    name="start",
                    showlegend=col == 1,
                ),
                row=1,
                col=col,
            )
        lock_aspect(fig, subplots=2)
        return STYLE.apply(fig, f"{self.title} · correlation {_correlation(learned, reference)}")


@dataclass(frozen=True)
class ValueMap(RepresentationPanel):
    name = "value_map"
    title = "Value: learned vs exact optimal"
    scope = "run"
    description = "Learned max_a F(s, a, z)·z next to the exact optimal goal-reaching value V*(s)."

    def options(self, runs: Sequence[RunData]) -> tuple[Option, ...]:
        return (_goal_choice(runs[0]),)

    def render(self, runs: Sequence[RunData], options: Mapping[str, Any]) -> Output:
        reps = _reps(runs[0])
        goal_cell = parse_cell(options["goal"])
        goal = reps.goal_index(goal_cell)
        learned = reps.q_values(goal).max(axis=1)
        oracle = build_goal_oracle(ENV.tabular_dynamics(), goal_cell, gamma=reps.gamma)
        exact = np.array([oracle.optimal.values[oracle.dynamics.index_of(s)] for s in reps.states])
        fig = make_subplots(
            rows=1, cols=2, subplot_titles=("learned max_a Q", "exact V*"), horizontal_spacing=0.12
        )
        for col, values in enumerate((learned, exact), start=1):
            cells = {state: float(v) for state, v in zip(reps.states, values, strict=True)}
            colorbar = {"thickness": 12, "x": 0.45 if col == 1 else 1.0}
            add_grid(
                fig,
                cells,
                row=1,
                col=col,
                zmin=float(values.min()),
                zmax=float(values.max()),
                colorbar=colorbar,
            )
            add_goal(fig, reps.goals[goal], row=1, col=col)
        lock_aspect(fig, subplots=2)
        non_goal = np.array([s != reps.goals[goal] for s in reps.states])
        correlation = _correlation(learned[non_goal], exact[non_goal])
        return STYLE.apply(fig, f"{self.title} · correlation {correlation}")
