"""Network panels: every layer of an agent's networks, from a saved ``network_features_v1``.

Create the artifact with ``aeris analysis features RUN``. The panels work on any
agent that exports its layers (the ``FeatureExtractor`` contract); the maps and
probe targets are Four Rooms'. Probe scores are always held out (5-fold).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any
from weakref import WeakKeyDictionary

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from aeris.analysis import network_analysis as na
from aeris.analysis.network_features import INTERPOLATION, LENGTH, TASK, NetworkFeatures
from aeris.analysis.state_labels import StateLabel, four_rooms_labels
from aeris.analysis.task_scores import score_policy, task_baseline
from aeris.dashboard.data import RunData
from aeris.dashboard.panels.base import (
    STYLE,
    Choice,
    Message,
    Number,
    Option,
    Output,
    Panel,
    series_colour,
)
from aeris.dashboard.panels.grid import COUNT_SCALE, ENV, INK, MUTED, add_grid, lock_aspect
from aeris.domain.task import TaskSpec
from aeris.environments.four_rooms import Position, room_of
from aeris.environments.four_rooms_tasks import task_suite

PROBE_NAMES = {"linear": "linear (ridge)", "kernel": "kernel ridge (RBF)", "knn": "5 nearest"}
SHOW = ("score", "above chance")
CLUSTERING = ("k-means", "Ward")
SWEEPS = ("length of z", "interpolation")
ALIGNMENT = ("linear CKA", "linear predictability (R²)")
_TASKS: dict[str, TaskSpec] = {task.task_id: task for task in task_suite("four_rooms_v1").tasks}
_DYNAMICS = ENV.tabular_dynamics()
_PROBE_CACHE: WeakKeyDictionary[NetworkFeatures, dict[tuple, na.ProbeResult]] = WeakKeyDictionary()


def _features(run: RunData) -> NetworkFeatures:
    assert run.features is not None
    return run.features


def _goal(features: NetworkFeatures) -> Position:
    """The reference condition's goal: the task the probes' task labels describe."""
    return ENV.goal_from_task(_TASKS[features.conditions[features.reference].tasks[0]])


def _labels(features: NetworkFeatures) -> tuple[StateLabel, ...]:
    return four_rooms_labels(ENV, _DYNAMICS, goal=_goal(features), gamma=features.gamma)


def _layer_choice(features: NetworkFeatures, default: str = "F.out", key: str = "layer") -> Choice:
    layers = (
        features.layers
        if key == "layer"
        else tuple(sorted(features.conditioned, key=features.layers.index))
    )
    label = "Layer" if key == "layer" else "Layer to follow"
    return Choice(key, label, layers, default if default in layers else layers[0])


def _condition_choice(features: NetworkFeatures) -> Choice:
    labels = tuple(c.label for c in features.conditions)
    return Choice("condition", "z (for F and Q)", labels, labels[features.reference])


def _activations(features: NetworkFeatures, options: Mapping[str, Any]) -> np.ndarray:
    return features.layer(options["layer"], features.condition_index(options["condition"]))


def _subtitle(features: NetworkFeatures, options: Mapping[str, Any]) -> str:
    layer = options["layer"]
    return layer if layer not in features.conditioned else f"{layer} · z = {options['condition']}"


def _probe(
    features: NetworkFeatures, layer: str, condition: int, label: StateLabel, kind: str
) -> na.ProbeResult:
    """Held-out probe results, cached per saved artifact (they take milliseconds each)."""
    cache = _PROBE_CACHE.setdefault(features, {})
    key = (layer, condition if layer in features.conditioned else None, label.name, kind)
    if key not in cache:
        cache[key] = na.probe(
            features.layer(layer, condition),
            label.values,
            kind=kind,
            categorical=label.categorical,
            accepted=label.accepted,
        )
    return cache[key]


def _discrete_scale(count: int) -> list[list[Any]]:
    """A stepwise colour scale for class indices 0..count-1 (zmin -0.5, zmax count - 0.5)."""
    scale: list[list[Any]] = []
    for index in range(count):
        colour = series_colour(index)
        scale += [[index / count, colour], [(index + 1) / count, colour]]
    return scale


def _cells(values: np.ndarray) -> dict[tuple[int, ...], float]:
    return {state: float(v) for state, v in zip(_DYNAMICS.states, values, strict=True)}


def _class_map(
    fig: go.Figure, classes: np.ndarray, names: Sequence[str], *, col: int, legend: bool
) -> None:
    """Cells coloured by class, with one legend entry per class present."""
    count = len(names)
    add_grid(
        fig,
        _cells(classes),
        row=1,
        col=col,
        zmin=-0.5,
        zmax=count - 0.5,
        colorscale=_discrete_scale(count),
    )
    fig.data[-1].showscale = False
    fig.data[-1].hovertemplate = "row %{y}, col %{x}<extra></extra>"
    if legend:
        for index in sorted(set(int(c) for c in classes)):
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="markers",
                    marker={"symbol": "square", "size": 12, "color": series_colour(index)},
                    name=names[index],
                ),
                row=1,
                col=col,
            )


@dataclass(frozen=True)
class NetworkPanel(Panel):
    """Shared requirement: a saved network-features artifact."""

    def missing(self, runs: Sequence[RunData]) -> str | None:
        absent = [run for run in runs if run.features is None]
        if absent:
            return f"No saved network features. Run: aeris analysis features {absent[0].root}"
        return None


@dataclass(frozen=True)
class LayerRanks(NetworkPanel):
    name = "layer_ranks"
    title = "Rank by layer"
    scope = "run"
    description = (
        "How many independent directions each layer uses across the states: effective rank "
        "and srank, and how fast its variance accumulates over principal components."
    )

    def options(self, runs: Sequence[RunData]) -> tuple[Option, ...]:
        return (_condition_choice(_features(runs[0])),)

    def render(self, runs: Sequence[RunData], options: Mapping[str, Any]) -> Output:
        features = _features(runs[0])
        condition = features.condition_index(options["condition"])
        layers = features.layers
        spectra = [na.singular_values(features.layer(name, condition)) for name in layers]
        ranks = [na.effective_rank(s) for s in spectra]
        sranks = [na.srank(s) for s in spectra]
        dead = [na.dead_units(features.layer(name, condition)) for name in layers]
        widths = [features.layer(name, condition).shape[1] for name in layers]
        fig = make_subplots(
            rows=1,
            cols=2,
            column_widths=[0.45, 0.55],
            subplot_titles=("rank of each layer", "share of variance in the first k components"),
        )
        networks = list(dict.fromkeys(name.split(".")[0] for name in layers))
        for index, network in enumerate(networks):
            members = [i for i, name in enumerate(layers) if name.split(".")[0] == network]
            hover = [
                f"{layers[i]}: {widths[i]} units, {dead[i]:.0%} never change<br>"
                f"effective rank {ranks[i]:.1f}, srank {sranks[i]}"
                for i in members
            ]
            for values, dash, measure in (
                (ranks, "solid", "effective rank"),
                (sranks, "dot", "srank"),
            ):
                fig.add_trace(
                    go.Scatter(
                        x=[layers[i] for i in members],
                        y=[values[i] for i in members],
                        mode="lines+markers",
                        line={"color": series_colour(index), "dash": dash, "width": 2},
                        marker={"size": 8},
                        name=f"{network} · {measure}",
                        text=hover,
                        hovertemplate="%{text}<extra></extra>",
                    ),
                    row=1,
                    col=1,
                )
        states = len(features.states)
        fig.add_hline(y=states - 1, line={"color": MUTED, "dash": "dot"}, row=1, col=1)
        fig.update_yaxes(title="dimensions", rangemode="tozero", row=1, col=1)
        fig.update_xaxes(categoryorder="array", categoryarray=list(layers), row=1, col=1)
        shares = np.array(
            [
                np.cumsum(s**2) / (s**2).sum() if (s**2).sum() > 0 else np.zeros(len(s))
                for s in (np.pad(s, (0, states - len(s))) for s in spectra)
            ]
        )
        fig.add_trace(
            go.Heatmap(
                z=shares,
                x=np.arange(1, states + 1),
                y=list(layers),
                zmin=0,
                zmax=1,
                colorscale=COUNT_SCALE,
                colorbar={"title": "share", "thickness": 12},
                hovertemplate="%{y}, first %{x} components: %{z:.0%}<extra></extra>",
            ),
            row=1,
            col=2,
        )
        fig.update_yaxes(autorange="reversed", row=1, col=2)
        fig.update_xaxes(title="components k", row=1, col=2)
        z = "" if not features.conditioned else f" · z = {options['condition']}"
        fig = STYLE.apply(
            fig,
            f"{self.title}<br><sup>{states} states: at most {states - 1} directions (dotted)"
            f"{z}</sup>",
        )
        fig.update_layout(margin={**STYLE.margin, "t": 90}, legend={"orientation": "h", "y": -0.25})
        return fig


@dataclass(frozen=True)
class FeaturePCA(NetworkPanel):
    name = "feature_pca"
    title = "Layer PCA"
    scope = "run"
    description = "Every state's activations in one layer, projected on its principal components."

    def options(self, runs: Sequence[RunData]) -> tuple[Option, ...]:
        features = _features(runs[0])
        colours = tuple(label.name for label in _labels(features) if label.scalar)
        return (
            _layer_choice(features),
            _condition_choice(features),
            Choice("colour", "Colour by", colours, "room"),
            Choice("dimensions", "Dimensions", (2, 3), 2),
        )

    def render(self, runs: Sequence[RunData], options: Mapping[str, Any]) -> Output:
        features = _features(runs[0])
        dims = int(options["dimensions"])
        scores, share = na.pca(_activations(features, options), dims)
        label = next(lab for lab in _labels(features) if lab.name == options["colour"])
        text = np.array([f"({r},{c}) {room_of((r, c))}" for r, c in features.states])
        fig = go.Figure()
        groups = (
            [(name, label.values == i) for i, name in enumerate(label.classes)]
            if label.categorical
            else [(label.name, np.ones(len(text), dtype=bool))]
        )
        for index, (name, mask) in enumerate(groups):
            if not mask.any():
                continue
            marker: dict[str, Any] = {"size": 8 if dims == 2 else 4, "line": {"width": 0}}
            if label.categorical:
                marker["color"] = series_colour(index)
            else:
                marker.update(
                    color=label.values,
                    colorscale=COUNT_SCALE,
                    showscale=True,
                    colorbar={"title": label.name, "thickness": 12},
                )
            common = {
                "mode": "markers",
                "marker": marker,
                "name": name,
                "text": list(text[mask]),
                "hovertemplate": "%{text}<extra></extra>",
            }
            points = scores[mask]
            if dims == 3:
                fig.add_trace(
                    go.Scatter3d(x=points[:, 0], y=points[:, 1], z=points[:, 2], **common)
                )
            else:
                fig.add_trace(go.Scatter(x=points[:, 0], y=points[:, 1], **common))
        if dims == 2:
            fig.update_xaxes(title=f"PC1 ({share[0]:.0%})")
            fig.update_yaxes(title=f"PC2 ({share[1]:.0%})")
        variance = ", ".join(f"PC{i + 1} {s:.0%}" for i, s in enumerate(share))
        fig = STYLE.apply(
            fig, f"{self.title} · {_subtitle(features, options)}<br><sup>{variance}</sup>"
        )
        fig.update_layout(margin={**STYLE.margin, "t": 80})
        return fig


@dataclass(frozen=True)
class ProbeMatrix(NetworkPanel):
    name = "probe_matrix"
    title = "What each layer encodes"
    scope = "run"
    description = (
        "Held-out probe score (5-fold) of every target from every layer: R² for values, "
        "accuracy for classes; 'above chance' rescales accuracy to (acc − chance)/(1 − chance)."
    )

    def options(self, runs: Sequence[RunData]) -> tuple[Option, ...]:
        return (
            _condition_choice(_features(runs[0])),
            Choice("probe", "Probe", na.PROBES, "linear"),
            Choice("show", "Show", SHOW, "score"),
        )

    def render(self, runs: Sequence[RunData], options: Mapping[str, Any]) -> Output:
        features = _features(runs[0])
        condition = features.condition_index(options["condition"])
        labels = _labels(features)
        layers = features.layers
        rows, values, text = [], [], []
        for label in labels:
            results = [
                _probe(features, layer, condition, label, options["probe"]) for layer in layers
            ]
            first = results[0]
            if label.categorical:
                rows.append(f"{label.name} · accuracy (chance {first.chance:.2f})")
            else:
                rows.append(f"{label.name} · R²")
            row = []
            for result in results:
                score = result.score
                if options["show"] == "above chance" and label.categorical:
                    score = (
                        (score - result.chance) / (1 - result.chance) if result.chance < 1 else 0.0
                    )
                row.append(score)
            values.append(row)
            text.append([f"{v:.2f}" if v > -1 else "< −1" for v in row])
        fig = go.Figure(
            go.Heatmap(
                z=np.clip(np.array(values), 0, 1),
                x=list(layers),
                y=rows,
                text=text,
                texttemplate="%{text}",
                textfont={"size": 11},
                zmin=0,
                zmax=1,
                colorscale=COUNT_SCALE,
                colorbar={"title": options["show"], "thickness": 12},
                hovertemplate="%{y}<br>%{x}: %{text}<extra></extra>",
            )
        )
        fig.update_yaxes(autorange="reversed")
        goal = _goal(features)
        z = f" · z = {options['condition']}" if features.conditioned else ""
        fig = STYLE.apply(
            fig,
            f"{self.title} · {PROBE_NAMES[options['probe']]} probe<br><sup>held out over 5 folds"
            f" of {len(features.states)} states · task targets for goal {goal}{z}</sup>",
        )
        fig.update_layout(
            height=max(STYLE.height, 60 + 34 * len(rows)), margin={**STYLE.margin, "t": 80}
        )
        return fig


@dataclass(frozen=True)
class ProbeMap(NetworkPanel):
    name = "probe_map"
    title = "Probe on the grid"
    scope = "run"
    description = "One probe's held-out prediction for every cell, next to the truth."

    def options(self, runs: Sequence[RunData]) -> tuple[Option, ...]:
        features = _features(runs[0])
        targets = tuple(label.name for label in _labels(features) if label.scalar)
        return (
            _layer_choice(features),
            _condition_choice(features),
            Choice("target", "Target", targets, "distance to goal"),
            Choice("probe", "Probe", na.PROBES, "linear"),
        )

    def render(self, runs: Sequence[RunData], options: Mapping[str, Any]) -> Output:
        features = _features(runs[0])
        label = next(lab for lab in _labels(features) if lab.name == options["target"])
        condition = features.condition_index(options["condition"])
        result = _probe(features, options["layer"], condition, label, options["probe"])
        fig = make_subplots(
            rows=1, cols=2, subplot_titles=("truth", "held-out prediction"), horizontal_spacing=0.06
        )
        if label.categorical:
            _class_map(fig, label.values, label.classes, col=1, legend=True)
            _class_map(fig, result.predictions, label.classes, col=2, legend=False)
            accepted = (
                label.accepted
                if label.accepted is not None
                else np.eye(len(label.classes), dtype=bool)[label.values]
            )
            wrong = [
                s
                for i, s in enumerate(features.states)
                if not accepted[i, int(result.predictions[i])]
            ]
            fig.add_trace(
                go.Scatter(
                    x=[s[1] for s in wrong],
                    y=[s[0] for s in wrong],
                    mode="text",
                    text=["×"] * len(wrong),
                    textfont={"color": "white", "size": 16},
                    name="wrong",
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=1,
                col=2,
            )
            summary = f"accuracy {result.score:.2f} (chance {result.chance:.2f}); × wrong"
        else:
            low, high = float(label.values.min()), float(label.values.max())
            for col, values in ((1, label.values), (2, result.predictions)):
                add_grid(
                    fig,
                    _cells(values),
                    row=1,
                    col=col,
                    zmin=low,
                    zmax=high,
                    colorscale=COUNT_SCALE,
                    colorbar={"title": label.name, "thickness": 12},
                )
                fig.data[-1].showscale = col == 2
            summary = f"R² {result.score:.2f}"
        lock_aspect(fig, 2)
        fig = STYLE.apply(
            fig,
            f"{label.name} from {_subtitle(features, options)}<br><sup>"
            f"{PROBE_NAMES[options['probe']]} probe · {summary}</sup>",
        )
        fig.update_layout(margin={**STYLE.margin, "t": 90})
        return fig


@dataclass(frozen=True)
class FeatureClusters(NetworkPanel):
    name = "feature_clusters"
    title = "Clusters of states"
    scope = "run"
    description = (
        "States grouped by their activations in one layer, painted on the grid; the adjusted "
        "Rand index says how well the groups match the rooms (1 exactly, 0 by chance)."
    )

    def options(self, runs: Sequence[RunData]) -> tuple[Option, ...]:
        features = _features(runs[0])
        return (
            _layer_choice(features),
            _condition_choice(features),
            Choice("method", "Method", CLUSTERING, "k-means"),
            Number("clusters", "Clusters", 4, minimum=2, maximum=8),
        )

    def render(self, runs: Sequence[RunData], options: Mapping[str, Any]) -> Output:
        features = _features(runs[0])
        x = _activations(features, options)
        k = int(options["clusters"])
        clusters = na.kmeans(x, k) if options["method"] == "k-means" else na.ward(x, k)
        labels = {label.name: label for label in _labels(features)}
        rooms = labels["room"].values
        walls = labels["next to a wall"].values
        fig = make_subplots(rows=1, cols=1)
        _class_map(fig, clusters, [f"cluster {i + 1}" for i in range(k)], col=1, legend=True)
        lock_aspect(fig)
        agreement = (
            f"agreement with rooms {na.adjusted_rand(clusters, rooms):.2f} · "
            f"with wall cells {na.adjusted_rand(clusters, walls):.2f} (adjusted Rand)"
        )
        fig = STYLE.apply(
            fig,
            f"{options['method']}, {k} clusters · {_subtitle(features, options)}"
            f"<br><sup>{agreement}</sup>",
        )
        fig.update_layout(margin={**STYLE.margin, "t": 80})
        return fig


def _edges() -> np.ndarray:
    """Every pair of neighbouring cells once [E, 2], as state indices."""
    found = {
        (min(s, int(t)), max(s, int(t)))
        for s in range(_DYNAMICS.num_states)
        for t in _DYNAMICS.next_states[s]
        if int(t) != s
    }
    return np.array(sorted(found))


@dataclass(frozen=True)
class StateSensitivity(NetworkPanel):
    name = "state_sensitivity"
    title = "Change between neighbouring cells"
    scope = "run"
    description = (
        "How far a layer's activations move with one step, drawn on each edge between "
        "neighbouring cells, relative to the median distance between any two states."
    )

    def options(self, runs: Sequence[RunData]) -> tuple[Option, ...]:
        features = _features(runs[0])
        return (_layer_choice(features), _condition_choice(features))

    def render(self, runs: Sequence[RunData], options: Mapping[str, Any]) -> Output:
        features = _features(runs[0])
        edges = _edges()
        distance = na.edge_distances(_activations(features, options), edges)
        states = _DYNAMICS.states
        middle = np.array([(np.array(states[a]) + np.array(states[b])) / 2 for a, b in edges])
        doorway = np.array(
            [room_of(states[a]) == "doorway" or room_of(states[b]) == "doorway" for a, b in edges]
        )
        fig = go.Figure()
        add_grid(fig)
        fig.add_trace(
            go.Scatter(
                x=middle[:, 1],
                y=middle[:, 0],
                mode="markers",
                marker={
                    "symbol": "square",
                    "size": 14,
                    "color": distance,
                    "colorscale": COUNT_SCALE,
                    "cmin": 0,
                    "showscale": True,
                    "colorbar": {"title": "step / median", "thickness": 12},
                    "line": {"width": 0},
                },
                text=[f"{states[a]} – {states[b]}" for a, b in edges],
                customdata=distance,
                hovertemplate="%{text}: %{customdata:.2f}<extra></extra>",
                showlegend=False,
            )
        )
        lock_aspect(fig)
        inside = float(np.median(distance[~doorway]))
        through = float(np.median(distance[doorway]))
        fig = STYLE.apply(
            fig,
            f"{self.title} · {_subtitle(features, options)}<br><sup>median step: "
            f"{inside:.2f} inside rooms, {through:.2f} through doorways "
            f"(1 = as far as two typical states)</sup>",
        )
        fig.update_layout(margin={**STYLE.margin, "t": 80})
        return fig


def _policy_score(actions: np.ndarray, task_id: str, gamma: float, cache: dict) -> float | None:
    task = _TASKS[task_id]
    rewards = ENV.state_rewards(task)
    if task_id not in cache:
        cache[task_id] = task_baseline(_DYNAMICS, rewards, gamma=gamma)
    return score_policy(_DYNAMICS, actions, rewards, cache[task_id], gamma=gamma).normalized_score


@dataclass(frozen=True)
class ZSensitivity(NetworkPanel):
    name = "z_sensitivity"
    title = "Sensitivity to z"
    scope = "run"
    description = (
        "What happens as the task vector changes: the reference direction B(goal) at other "
        "lengths, or a straight line to another goal's z. Actions changed, exact score, and how "
        "far the layer moves."
    )

    def options(self, runs: Sequence[RunData]) -> tuple[Option, ...]:
        features = _features(runs[0])
        return (
            Choice("sweep", "Vary", SWEEPS, SWEEPS[0]),
            _layer_choice(features, key="drift_layer"),
        )

    def render(self, runs: Sequence[RunData], options: Mapping[str, Any]) -> Output:
        features = _features(runs[0])
        reference = features.reference
        kind = LENGTH if options["sweep"] == SWEEPS[0] else INTERPOLATION
        path = sorted(
            (i for i, c in enumerate(features.conditions) if c.kind == kind),
            key=lambda i: features.conditions[i].position or 0.0,
        )
        if not path:
            return Message(f"The export has no {options['sweep']} conditions.")
        if "Q" not in features.features:
            return Message("The export has no Q layer, so no policies to compare.")

        def actions(condition: int) -> np.ndarray:
            found = features.actions(condition)
            assert found is not None
            return found

        x = [features.conditions[i].position for i in path]
        layer = options["drift_layer"]
        ends = [path[0], path[-1]] if kind == INTERPOLATION else [reference]
        tasks = (
            features.conditions[path[-1]].tasks
            if kind == INTERPOLATION
            else features.conditions[reference].tasks
        )
        cache: dict = {}
        fig = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=(
                "actions that match"
                + (" each end's policy" if kind == INTERPOLATION else " B(goal)'s policy"),
                "exact normalized score",
                f"change in {layer} (‖Δ‖ / ‖reference‖)",
            ),
        )
        for index, end in enumerate(ends):
            name = features.conditions[end].label
            same = [float(np.mean(actions(i) == actions(end))) for i in path]
            change = [
                na.relative_change(features.layer(layer, i), features.layer(layer, end))
                for i in path
            ]
            for row, values in ((1, same), (3, change)):
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=values,
                        mode="lines+markers",
                        line={"color": series_colour(index), "width": 2},
                        name=f"vs {name}",
                        legendgroup=f"end{index}",
                        showlegend=row == 1,
                        hovertemplate="%{x:g}: %{y:.2f}<extra></extra>",
                    ),
                    row=row,
                    col=1,
                )
        for index, task_id in enumerate(tasks):
            scores = [_policy_score(actions(i), task_id, features.gamma, cache) for i in path]
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=scores,
                    mode="lines+markers",
                    line={"color": series_colour(len(ends) + index), "width": 2},
                    name=f"score on {task_id}",
                    hovertemplate="%{x:g}: %{y:.3f}<extra></extra>",
                ),
                row=2,
                col=1,
            )
        if kind == LENGTH:
            inferred = [
                i
                for i, c in enumerate(features.conditions)
                if c.kind == TASK and c.tasks == features.conditions[reference].tasks
            ]
            marks = [(features.conditions[reference].length, "B(goal)")]
            marks += [(features.conditions[i].length, "inferred z") for i in inferred]
            for length, text in marks:
                fig.add_vline(x=length, line={"color": MUTED, "dash": "dot"})
                fig.add_annotation(
                    x=np.log10(length),  # annotations on a log axis take log10 positions
                    y=0.04,
                    yref="y domain",  # inside the top chart, by its bottom edge
                    yanchor="bottom",
                    xanchor="left",
                    text=f" {text}",
                    showarrow=False,
                    font={"color": INK, "size": 11},
                )
            fig.update_xaxes(type="log")
            fig.update_xaxes(title="|z|", row=3, col=1)
        else:
            fig.update_xaxes(title="t (0 = B(goal), 1 = the other goal's B)", row=3, col=1)
        fig.update_yaxes(range=[0, 1.05], row=1, col=1)
        fig.update_yaxes(rangemode="tozero", row=3, col=1)
        fig = STYLE.apply(fig, f"{self.title} · {options['sweep']}")
        fig.update_layout(height=640, margin={**STYLE.margin, "t": 70})
        return fig


@dataclass(frozen=True)
class LayerAlignment(NetworkPanel):
    name = "layer_alignment"
    title = "Same layer across runs"
    scope = "runs"
    description = (
        "How alike one layer is between runs (seeds, datasets): linear CKA (same geometry up "
        "to rotation and scale), or how well a linear map from the row run predicts the "
        "column run (held out; invariant to any invertible linear map)."
    )

    def options(self, runs: Sequence[RunData]) -> tuple[Option, ...]:
        features = _features(runs[0])
        common = [
            c.label
            for c in features.conditions
            if all(c.label in {d.label for d in _features(r).conditions} for r in runs)
        ]
        default = features.conditions[features.reference].label
        return (
            _layer_choice(features),
            Choice(
                "condition",
                "z (for F and Q)",
                tuple(common),
                default if default in common else common[0],
            ),
            Choice("measure", "Measure", ALIGNMENT, ALIGNMENT[0]),
        )

    def render(self, runs: Sequence[RunData], options: Mapping[str, Any]) -> Output:
        if len(runs) < 2:
            return Message("Select at least two runs to compare a layer across them.")
        layer = options["layer"]
        activations = [
            _features(run).layer(layer, _features(run).condition_index(options["condition"]))
            for run in runs
        ]
        n = len(runs)
        matrix = np.ones((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if options["measure"] == ALIGNMENT[0]:
                    matrix[i, j] = na.linear_cka(activations[i], activations[j])
                else:
                    matrix[i, j] = na.probe(activations[i], activations[j], kind="linear").score
        names = [run.label for run in runs]
        off = matrix[~np.eye(n, dtype=bool)]
        fig = go.Figure(
            go.Heatmap(
                z=np.clip(matrix, 0, 1),
                x=names,
                y=names,
                text=[[f"{v:.2f}" for v in row] for row in matrix],
                texttemplate="%{text}",
                textfont={"size": 10},
                zmin=0,
                zmax=1,
                colorscale=COUNT_SCALE,
                colorbar={"thickness": 12},
                hovertemplate="%{y} → %{x}: %{text}<extra></extra>",
            )
        )
        fig.update_yaxes(autorange="reversed")
        z = f" · z = {options['condition']}" if layer in _features(runs[0]).conditioned else ""
        fig = STYLE.apply(
            fig,
            f"{layer} across runs · {options['measure']}<br><sup>mean between runs "
            f"{off.mean():.2f}{z}</sup>",
        )
        fig.update_layout(height=max(STYLE.height, 120 + 32 * n), margin={**STYLE.margin, "t": 80})
        return fig
