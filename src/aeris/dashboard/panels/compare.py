"""Comparing groups of runs seed by seed: every run's value next to its group's mean."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from aeris.analysis.seeds import difference, mean_interval
from aeris.dashboard.data import RunData
from aeris.dashboard.panels.base import (
    STYLE,
    Choice,
    Message,
    MultiChoice,
    Option,
    Output,
    Panel,
    series_colour,
)
from aeris.dashboard.panels.grid import MUTED
from aeris.dashboard.panels.grouping import group_choice, group_runs
from aeris.dashboard.panels.tables import interval_text, metric_order, run_metrics

# Shown by default when present: the training goal's exact results, the representation's
# fit, and the zero-shot mean over all tasks per evidence protocol.
HEADLINE = (
    "final policy · all-start success",
    "final policy · optimal actions",
    "FB · occupancy correlation",
)


def _title(name: str) -> str:
    _, section, metric = metric_order(name)
    return f"{section} · {metric}"


def _metric_names(runs: Sequence[RunData]) -> dict[str, str]:
    """Every headline number of the runs, by display title."""
    return {_title(name): name for name in run_metrics(runs)}


@dataclass(frozen=True)
class SeedDots(Panel):
    name = "seed_dots"
    title = "Results per seed"
    scope = "runs"
    description = (
        "One row per metric: every run as a dot coloured by its group, with the group mean "
        "and its 95% interval; the title gives the difference from the first group."
    )

    def options(self, runs: Sequence[RunData]) -> tuple[Option, ...]:
        choice = group_choice(runs)
        names = _metric_names(runs)
        default = tuple(
            title
            for title, name in names.items()
            if name in HEADLINE or name.startswith("zero-shot · all · ")
        )
        return (
            Choice(choice.key, choice.label, choice.choices, "name"),
            MultiChoice("metrics", "Metrics", tuple(names), default),
        )

    def render(self, runs: Sequence[RunData], options: Mapping[str, Any]) -> Output:
        names = _metric_names(runs)
        chosen = [names[t] for t in names if t in options["metrics"]]
        if not chosen:
            return Message("Choose at least one metric.")
        groups = group_runs(runs, options["group_by"])
        values = {group: run_metrics(members) for group, members in groups.items()}
        first = next(iter(groups))
        titles = []
        for name in chosen:
            gaps = [
                f"{group} − {first} {text}"
                for group in list(groups)[1:]
                if (
                    text := interval_text(
                        difference(values[first].get(name, []), values[group].get(name, []))
                    )
                )
            ]
            titles.append(" · ".join([_title(name), *gaps]))
        fig = make_subplots(
            rows=len(chosen), cols=1, subplot_titles=titles, vertical_spacing=0.5 / len(chosen)
        )
        count = len(groups)
        for row, name in enumerate(chosen, start=1):
            for index, (group, members) in enumerate(groups.items()):
                colour = series_colour(index)
                y = count - 1 - index  # the first group on top
                found = values[group].get(name, [None] * len(members))
                seeds = [(run, v) for run, v in zip(members, found, strict=True) if v is not None]
                if not seeds:
                    fig.add_annotation(
                        x=0.5,
                        xref=f"x{row} domain",
                        y=y,
                        yref=f"y{row}",
                        text="no report",
                        showarrow=False,
                        font={"color": MUTED},
                    )
                    continue
                spread = [0.12 + 0.16 * (i / max(len(seeds) - 1, 1)) for i in range(len(seeds))]
                fig.add_trace(
                    go.Scatter(
                        x=[v for _, v in seeds],
                        y=[y + offset for offset in spread],
                        mode="markers",
                        marker={"color": colour, "size": 9, "line": {"color": "white", "width": 1}},
                        name=group,
                        legendgroup=group,
                        showlegend=row == 1,
                        text=[run.label for run, _ in seeds],
                        hovertemplate="%{text}: %{x:.3f}<extra></extra>",
                    ),
                    row=row,
                    col=1,
                )
                interval = mean_interval(v for _, v in seeds)
                mean = interval.value if interval else seeds[0][1]
                error = (
                    {
                        "type": "data",
                        "symmetric": False,
                        "array": [interval.high - mean],
                        "arrayminus": [mean - interval.low],
                        "color": colour,
                        "thickness": 2,
                        "width": 0,
                    }
                    if interval
                    else None
                )
                fig.add_trace(
                    go.Scatter(
                        x=[mean],
                        y=[y - 0.15],
                        mode="markers",
                        marker={"color": colour, "size": 11, "symbol": "diamond"},
                        error_x=error,
                        name=f"{group} mean",
                        legendgroup=group,
                        showlegend=False,
                        hovertemplate=(
                            f"{group}: mean %{{x:.3f}}"
                            + (f" [{interval.low:.3f}, {interval.high:.3f}]" if interval else "")
                            + "<extra></extra>"
                        ),
                    ),
                    row=row,
                    col=1,
                )
            fig.update_yaxes(
                row=row,
                col=1,
                range=[-0.5, count - 0.5],
                tickvals=list(range(count)),
                ticktext=list(reversed(groups)),
                showgrid=False,
            )
        legend = "dots: runs · diamond and bar: group mean with its 95% interval"
        fig = STYLE.apply(fig, f"{self.title}<br><sup>{legend}</sup>")
        fig.update_layout(
            height=max(STYLE.height, 80 + (70 + 45 * count) * len(chosen)),
            margin={**STYLE.margin, "t": 90},
        )
        return fig
