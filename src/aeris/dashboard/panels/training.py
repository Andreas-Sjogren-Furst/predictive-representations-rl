"""Training curves for one or many runs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import plotly.graph_objects as go

from aeris.dashboard.data import RunData
from aeris.dashboard.panels.base import STYLE, Message, MultiChoice, Number, Option, Output, Panel


def _metric_names(runs: Sequence[RunData]) -> tuple[str, ...]:
    names: dict[str, None] = {}
    for run in runs:
        for record in run.training:
            for metric in record.metrics:
                names.setdefault(metric.name)
    return tuple(names)


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
    description = "Any logged training metric over gradient updates; overlays every selected run."

    def options(self, runs: Sequence[RunData]) -> tuple[Option, ...]:
        names = _metric_names(runs)
        default = tuple(n for n in ("fb_loss", "fb_regularization") if n in names) or names[:1]
        return (
            MultiChoice("metrics", "Metrics", names, default),
            Number("smoothing", "Smoothing (logged points)", 1, minimum=1, maximum=200),
        )

    def missing(self, runs: Sequence[RunData]) -> str | None:
        return None if any(run.training for run in runs) else "No training metrics recorded yet."

    def render(self, runs: Sequence[RunData], options: Mapping[str, Any]) -> Output:
        metrics = options["metrics"]
        if not metrics:
            return Message("Choose at least one metric.")
        fig = go.Figure()
        for run in runs:
            for metric in metrics:
                points = [
                    (record.step, m.value)
                    for record in run.training
                    for m in record.metrics
                    if m.name == metric
                ]
                if not points:
                    continue
                steps, values = (np.asarray(v, dtype=float) for v in zip(*points, strict=True))
                label = metric if len(runs) == 1 else f"{run.label} · {metric}"
                fig.add_trace(
                    go.Scatter(
                        x=steps,
                        y=smooth(values, int(options["smoothing"])),
                        mode="lines",
                        name=label,
                    )
                )
        fig.update_xaxes(title="gradient updates")
        return STYLE.apply(fig, self.title)
