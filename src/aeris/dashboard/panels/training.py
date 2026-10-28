"""Training curves for one or many runs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from aeris.dashboard.data import RunData
from aeris.dashboard.panels.base import (
    STYLE,
    Message,
    MultiChoice,
    Number,
    Option,
    Output,
    Panel,
    Toggle,
    series_colour,
)
from aeris.dashboard.panels.grouping import group_choice, group_runs, label, mean_and_std


def _metric_names(runs: Sequence[RunData]) -> tuple[str, ...]:
    names: dict[str, None] = {}
    for run in runs:
        for record in run.training:
            for metric in record.metrics:
                names.setdefault(metric.name)
    return tuple(names)


def _series(run: RunData, metric: str, window: int) -> tuple[np.ndarray, np.ndarray] | None:
    points = [(r.step, m.value) for r in run.training for m in r.metrics if m.name == metric]
    if not points:
        return None
    steps, values = (np.asarray(v, dtype=float) for v in zip(*points, strict=True))
    return steps, smooth(values, int(window))


def smooth(values: np.ndarray, window: int) -> np.ndarray:
    """Trailing moving average over ``window`` logged points (1 = raw values)."""
    if window <= 1 or len(values) < 2:
        return values
    cumulative = np.cumsum(np.insert(values, 0, 0.0))
    out = np.empty_like(values)
    for i in range(len(values)):
        start = max(0, i + 1 - window)
        out[i] = (cumulative[i + 1] - cumulative[start]) / (i + 1 - start)
    return out


@dataclass(frozen=True)
class LossCurves(Panel):
    name = "loss_curves"
    title = "Training curves"
    scope = "runs"
    description = (
        "Any logged metric over updates, one chart per metric; overlays runs or groups "
        "(mean ± std band)."
    )

    def options(self, runs: Sequence[RunData]) -> tuple[Option, ...]:
        names = _metric_names(runs)
        default = tuple(n for n in ("fb_loss", "fb_regularization") if n in names) or names[:1]
        return (
            MultiChoice("metrics", "Metrics", names, default),
            Number("smoothing", "Smoothing (logged points)", 1, minimum=1, maximum=200),
            Toggle("separate", "One chart per metric", True),
            group_choice(runs),
        )

    def missing(self, runs: Sequence[RunData]) -> str | None:
        return None if any(run.training for run in runs) else "No training metrics recorded yet."

    def render(self, runs: Sequence[RunData], options: Mapping[str, Any]) -> Output:
        metrics = options["metrics"]
        if not metrics:
            return Message("Choose at least one metric.")
        separate = options["separate"] and len(metrics) > 1
        fig = make_subplots(
            rows=len(metrics) if separate else 1,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.06,
            subplot_titles=metrics if separate else None,
        )
        groups = group_runs(runs, options["group_by"])
        single = len(groups) == 1 and len(next(iter(groups.values()))) == 1
        for row, metric in enumerate(metrics, start=1):
            for index, (group, members) in enumerate(groups.items()):
                series = [s for run in members if (s := _series(run, metric, options["smoothing"]))]
                if not series:
                    continue
                if separate:  # a group keeps its colour on every chart; one legend entry
                    name, colour = label(group, members), series_colour(index)
                    legend = row == 1 and not single
                else:
                    trace = (row - 1) * len(groups) + index
                    name = metric if single else f"{label(group, members)} · {metric}"
                    colour, legend = series_colour(trace), True
                place = {"row": row if separate else 1, "col": 1}
                steps, mean, std = mean_and_std(series)
                fig.add_trace(
                    go.Scatter(
                        x=steps,
                        y=mean,
                        mode="lines",
                        name=name,
                        legendgroup=name,
                        showlegend=legend,
                        line={"color": colour, "width": 2},
                    ),
                    **place,
                )
                if len(series) > 1:
                    fig.add_trace(
                        go.Scatter(
                            x=np.concatenate([steps, steps[::-1]]),
                            y=np.concatenate([mean + std, (mean - std)[::-1]]),
                            fill="toself",
                            fillcolor=colour,
                            opacity=0.2,
                            line={"width": 0},
                            hoverinfo="skip",
                            name=f"{name} ± std",
                            legendgroup=name,
                            showlegend=False,
                        ),
                        **place,
                    )
        rows = len(metrics) if separate else 1
        fig.update_xaxes(title="gradient updates", row=rows, col=1)
        fig = STYLE.apply(fig, self.title)
        if separate:
            fig.update_layout(height=max(STYLE.height, 60 + 170 * rows))
        return fig
