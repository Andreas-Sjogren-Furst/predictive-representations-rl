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
from aeris.analysis.tabular_occupancy import centered_pearson
from aeris.analysis.task_scores import optimal_solution
from aeris.dashboard.data import RunData
from aeris.dashboard.panels.base import (
    STYLE,
    Choice,
    Option,
    Output,
    Panel,
    Toggle,
    series_colour,
)
from aeris.dashboard.panels.behaviour import add_actions
from aeris.dashboard.panels.grid import (
    ARROWS,
    COUNT_SCALE,
    ENV,
    INK,
    SCORE_SCALE,
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
WORST_CELL = "costliest mistake"  # the cell where the learned choice loses the most value


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
            for index, room in enumerate(ROOM_ORDER):
                mask = np.array([room_of(s) == room for s in reps.states])
                if mask.any():
                    trace = self._scatter(coords[mask], np.array(labels)[mask], room, None)
                    trace.marker.color = series_colour(index)
                    fig.add_trace(trace)
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
                color=colour, colorscale=COUNT_SCALE, showscale=True, colorbar={"thickness": 12}
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
        names = [f"{cell_label(reps.states[i])} {room_of(reps.states[i])}" for i in order]
        rooms = [room_of(reps.states[i]) for i in order]
        limit = float(np.abs(similarity).max())
        fig = go.Figure(
            go.Heatmap(
                z=similarity,
                text=[[f"{a} vs {b}" for b in names] for a in names],
                colorscale=SCORE_SCALE,
                zmin=-limit,
                zmax=limit,
                hovertemplate="%{text}: %{z:.2f}<extra></extra>",
                colorbar={"thickness": 12},
            )
        )
        # Label each room's block at its middle, and draw thin lines between blocks.
        blocks = [room for room in ROOM_ORDER if room in rooms]
        middles = [rooms.index(room) + (rooms.count(room) - 1) / 2 for room in blocks]
        axis = {"tickvals": middles, "ticktext": blocks, "showgrid": False, "zeroline": False}
        fig.update_xaxes(**axis)
        fig.update_yaxes(autorange="reversed", **axis)
        for room in blocks[1:]:
            edge = rooms.index(room) - 0.5
            fig.add_hline(y=edge, line={"color": "white", "width": 1})
            fig.add_vline(x=edge, line={"color": "white", "width": 1})
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
        states, d = reps.backward.shape
        # Centring removes one direction: S states span at most S - 1, whatever d is.
        possible = min(d, states - 1)
        singular = np.linalg.svd(reps.backward - reps.backward.mean(axis=0), compute_uv=False)
        singular = singular[:possible]
        energy = singular**2 / (singular**2).sum()
        positive = energy[energy > 0]
        effective_rank = float(np.exp(-(positive * np.log(positive)).sum()))
        fig = go.Figure(
            go.Bar(
                x=np.arange(1, len(singular) + 1),
                y=singular,
                name="singular value",
                marker_color=series_colour(0),
            )
        )
        fig.update_xaxes(title="component")
        fig.update_yaxes(title="singular value", type="log" if options["log"] else "linear")
        bound = f"at most {possible} ({states} states − 1, d = {d})"
        return STYLE.apply(fig, f"{self.title} · effective rank {effective_rank:.1f} of {bound}")


def _hide_scale(fig: go.Figure) -> None:
    """Hide the colour bar of the heatmap just added (it repeats its neighbour's)."""
    fig.data[-1].showscale = False


def _ring(
    fig: go.Figure, cell: tuple[int, ...], *, col: int, name: str = "start", size: int = 18
) -> None:
    fig.add_trace(
        go.Scatter(
            x=[cell[1]],
            y=[cell[0]],
            mode="markers",
            marker={"symbol": "circle-open", "size": size, "color": INK, "line": {"width": 3}},
            name=name,
            showlegend=col == 1,
            hoverinfo="skip",
        ),
        row=1,
        col=col,
    )


def _mass(values: np.ndarray, density: np.ndarray) -> float:
    """Occupancy mass of a density m/ρ: the sum of m = (m/ρ)·ρ over visited states."""
    keep = np.isfinite(values)
    return float((values[keep] * density[keep]).sum())


@dataclass(frozen=True)
class OccupancyMaps(RepresentationPanel):
    name = "occupancy_maps"
    title = "Occupancy: learned vs exact"
    scope = "run"
    description = (
        "Learned F(s0, a, z)·B(t) next to the exact occupancy density m/ρ from one start, and "
        "their difference: where the learned occupancy is missing mass (red) or has too much."
    )

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
        density = reps.state_density.astype(np.float64)
        share = _mass(learned, density) / max(_mass(reference, density), 1e-12)
        correlation = _correlation(learned, reference)
        if options["log"]:
            learned = np.log10(np.clip(learned, 1e-3, None))
            reference = np.log10(np.clip(reference, 1e-3, None))
        difference = learned - reference
        finite = np.concatenate([learned[np.isfinite(learned)], reference[np.isfinite(reference)]])
        low, high = float(finite.min()), float(finite.max())
        unit = "log10 m/ρ" if options["log"] else "m/ρ"
        fig = make_subplots(
            rows=1,
            cols=3,
            subplot_titles=("learned F·B", "exact m/ρ", "learned − exact"),
            horizontal_spacing=0.07,
        )
        for col, values in enumerate((learned, reference), start=1):
            cells = {state: float(v) for state, v in zip(reps.states, values, strict=True)}
            add_grid(
                fig,
                cells,
                row=1,
                col=col,
                zmin=low,
                zmax=high,
                colorscale=COUNT_SCALE,
                colorbar={"thickness": 12, "title": unit, "x": 0.64},
            )
            if col == 1:
                _hide_scale(fig)
        bound = float(np.nanmax(np.abs(difference))) or 1.0
        cells = {state: float(v) for state, v in zip(reps.states, difference, strict=True)}
        add_grid(
            fig,
            cells,
            row=1,
            col=3,
            zmin=-bound,
            zmax=bound,
            colorscale=SCORE_SCALE,
            colorbar={"thickness": 12, "title": "difference", "x": 1.0},
        )
        for col in (1, 2, 3):
            add_goal(fig, reps.goals[goal], row=1, col=col)
            _ring(fig, reps.states[start], col=col)
        lock_aspect(fig, subplots=3)
        fig = STYLE.apply(
            fig,
            f"{self.title} · correlation {correlation} · learned mass {share:.0%} of exact",
        )
        fig.update_layout(legend={"orientation": "h", "x": 0, "y": -0.06})
        return fig


def _goal_values(reps: FBRepresentations, goal: int) -> tuple[np.ndarray, ...]:
    """Learned and exact Q(s, a) for z = B(goal), all in FB's units m(s, a, goal)/ρ(goal).

    Learned: F(s, a, z)·z. Exact, same policy: the occupancy of the goal under FB's own target
    policy. Exact, optimal: the best possible occupancy of the goal (continuing: reward 1 in
    the goal, the current state included, nothing terminal). Each is [S, A].
    """
    cell = reps.goals[goal]
    index = reps.state_index(cell)
    dynamics = ENV.tabular_dynamics()
    order = [dynamics.index_of(state) for state in reps.states]
    rewards = np.zeros(dynamics.num_states)
    rewards[dynamics.index_of(cell)] = 1.0
    optimal = optimal_solution(dynamics, rewards, gamma=reps.gamma)
    rho = float(reps.state_density[index])
    scale = 1.0 / rho if rho > 0 else np.nan
    return (
        reps.q_values(goal).astype(np.float64),
        reps.exact_density[goal][:, :, index].astype(np.float64),
        optimal.action_values[order] * scale,
        optimal.optimal_actions[order],
    )


REFERENCES = ("same policy", "optimal policy")


@dataclass(frozen=True)
class ValueMap(RepresentationPanel):
    name = "value_map"
    title = "Value: learned vs exact"
    scope = "run"
    description = (
        "Learned value F·z of z = B(goal) next to the exact value of the same objective (the "
        "goal's discounted occupancy over ρ): under FB's own policy, or under an optimal one."
    )

    def options(self, runs: Sequence[RunData]) -> tuple[Option, ...]:
        return (
            _goal_choice(runs[0]),
            Choice("reference", "Exact reference", REFERENCES, REFERENCES[0]),
        )

    def render(self, runs: Sequence[RunData], options: Mapping[str, Any]) -> Output:
        reps = _reps(runs[0])
        goal = reps.goal_index(parse_cell(options["goal"]))
        learned_q, on_policy, optimal, _ = _goal_values(reps, goal)
        if options["reference"] == "same policy":
            policy = reps.target_policy[goal].astype(np.float64)
            learned = (policy * learned_q).sum(axis=1)
            exact = (policy * on_policy).sum(axis=1)
            names = ("learned V^π = Σ π F·z", "exact V^π")
        else:
            learned, exact = learned_q.max(axis=1), optimal.max(axis=1)
            names = ("learned max_a F·z", "exact V*")
        finite = np.concatenate([learned[np.isfinite(learned)], exact[np.isfinite(exact)]])
        low, high = float(finite.min()), float(finite.max())
        fig = make_subplots(rows=1, cols=2, subplot_titles=names, horizontal_spacing=0.08)
        for col, values in enumerate((learned, exact), start=1):
            cells = {state: float(v) for state, v in zip(reps.states, values, strict=True)}
            add_grid(
                fig,
                cells,
                row=1,
                col=col,
                zmin=low,
                zmax=high,
                colorscale=COUNT_SCALE,
                colorbar={"thickness": 12, "title": "m/ρ of goal"},
            )
            if col == 1:
                _hide_scale(fig)
            add_goal(fig, reps.goals[goal], row=1, col=col)
        lock_aspect(fig, subplots=2)
        non_goal = np.array([s != reps.goals[goal] for s in reps.states])
        correlation = _correlation(learned[non_goal], exact[non_goal])
        keep = non_goal & np.isfinite(exact)
        ratio = learned[keep].sum() / max(exact[keep].sum(), 1e-12)
        fig = STYLE.apply(
            fig,
            f"{self.title} ({options['reference']}) · correlation {correlation}"
            f" · learned is {ratio:.0%} of exact",
        )
        fig.update_layout(legend={"orientation": "h", "x": 0, "y": -0.06})
        return fig


@dataclass(frozen=True)
class ActionValues(RepresentationPanel):
    name = "action_values"
    title = "Action values"
    scope = "run"
    description = (
        "Why FB picks an action, for z = B(goal): a map of the cells where its best learned "
        "action is not optimal, and the learned and exact value of each action in one cell."
    )

    def options(self, runs: Sequence[RunData]) -> tuple[Option, ...]:
        reps = _reps(runs[0])
        cells = (WORST_CELL, *(cell_label(s) for s in reps.states))
        return (_goal_choice(runs[0]), Choice("cell", "Cell", cells, WORST_CELL))

    def render(self, runs: Sequence[RunData], options: Mapping[str, Any]) -> Output:
        reps = _reps(runs[0])
        goal = reps.goal_index(parse_cell(options["goal"]))
        learned, on_policy, optimal, best = _goal_values(reps, goal)
        greedy = learned.argmax(axis=1)
        states = np.arange(len(reps.states))
        goal_state = reps.state_index(reps.goals[goal])
        right = best[states, greedy]
        right[goal_state] = True  # every action is as good at the goal; not a mistake
        # What the learned choice costs, as a share of the best action's value there.
        loss = 1 - optimal[states, greedy] / np.maximum(optimal.max(axis=1), 1e-12)
        if options["cell"] == WORST_CELL:
            cell = int(np.argmax(np.where(right, -np.inf, loss))) if (~right).any() else 0
        else:
            cell = reps.state_index(parse_cell(options["cell"]))
        fig = make_subplots(
            rows=1,
            cols=2,
            column_widths=[0.55, 0.45],
            subplot_titles=("best learned action", f"actions in {cell_label(reps.states[cell])}"),
            horizontal_spacing=0.1,
        )
        add_grid(fig, row=1, col=1)
        # In the dynamics' state order, as the shared map helpers expect.
        dynamics = ENV.tabular_dynamics()
        actions = np.empty(dynamics.num_states, dtype=np.int64)
        mask = np.ones((dynamics.num_states, len(ARROWS)), dtype=bool)
        for index, state in enumerate(reps.states):
            actions[dynamics.index_of(state)] = greedy[index]
            mask[dynamics.index_of(state)] = best[index]
        add_actions(fig, dynamics, actions, mask, skip=dynamics.index_of(reps.goals[goal]))
        add_goal(fig, reps.goals[goal], row=1, col=1)
        _ring(fig, reps.states[cell], col=1, name="cell", size=28)  # clear of a marked arrow
        series = (
            ("learned F·z", learned),
            ("exact, same policy", on_policy),
            ("exact, optimal", optimal),
        )
        for index, (name, values) in enumerate(series):
            row = values[cell]
            relative = row / max(float(np.nanmax(row)), 1e-12)  # the scales differ by up to ~50×
            fig.add_trace(
                go.Bar(
                    x=list(ARROWS.values()),
                    y=relative,
                    customdata=row,
                    name=name,
                    marker_color=series_colour(index),
                    hovertemplate=(
                        f"{name} %{{x}}: %{{y:.0%}} of the best ({'%{customdata:.3g}'} m/ρ)"
                        "<extra></extra>"
                    ),
                ),
                row=1,
                col=2,
            )
        fig.update_yaxes(title="value / best action's value", tickformat=".0%", row=1, col=2)
        lock_aspect(fig)
        fig.update_layout(barmode="group", legend={"orientation": "h", "x": 0, "y": -0.08})
        verdict = "optimal" if right[cell] else f"not optimal ({loss[cell]:.1%} below the best)"
        summary = (
            f"optimal in {right[states != goal_state].mean():.0%} of cells · "
            f"{cell_label(reps.states[cell])}: learned {ARROWS[int(greedy[cell])]}, {verdict}"
        )
        title = f"{self.title} · goal {cell_label(reps.goals[goal])}"
        fig = STYLE.apply(fig, f"{title}<br><sup>{summary}</sup>")
        fig.update_layout(margin={**STYLE.margin, "t": 90})  # room for the summary line
        return fig
