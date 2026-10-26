"""Recorded evaluation results per checkpoint."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import plotly.graph_objects as go

from aeris.dashboard.data import RunData
from aeris.dashboard.panels.base import STYLE, Choice, Option, Output, Panel


@dataclass(frozen=True)
class EvaluationResults(Panel):
    name = "evaluation"
    title = "Evaluation"
    scope = "runs"
    description = "Recorded evaluation metrics for each evaluated checkpoint, per run."

    def options(self, runs: Sequence[RunData]) -> tuple[Option, ...]:
        names = tuple(
            dict.fromkeys(m.name for run in runs for e in run.evaluations for m in e.metrics)
        )
        return (
            Choice("metric", "Metric", names, "success_rate" if "success_rate" in names else None),
        )

    def missing(self, runs: Sequence[RunData]) -> str | None:
        return None if any(run.evaluations for run in runs) else "No evaluations recorded."

    def render(self, runs: Sequence[RunData], options: Mapping[str, Any]) -> Output:
        metric = options["metric"]
        fig = go.Figure()
        for run in runs:
            pairs = [
                (record.checkpoint_id, m.value)
                for record in run.evaluations
                for m in record.metrics
                if m.name == metric
            ]
            if pairs:
                checkpoints, values = zip(*pairs, strict=True)
                fig.add_trace(go.Bar(x=list(checkpoints), y=list(values), name=run.label))
        fig.update_layout(barmode="group")
        fig.update_yaxes(title=metric)
        return STYLE.apply(fig, f"{self.title} · {metric}")
