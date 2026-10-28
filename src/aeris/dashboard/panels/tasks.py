"""Zero-shot task-suite panels, drawn from a saved ``zero_shot_suite_v1`` report.

Create the report with ``aeris analysis zero-shot RUN``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import plotly.graph_objects as go

from aeris.analysis.seeds import summarize
from aeris.dashboard.data import RunData
from aeris.dashboard.panels.base import (
    STYLE,
    Choice,
    Message,
    Option,
    Output,
    Panel,
    Table,
    series_colour,
)
from aeris.dashboard.panels.grid import COUNT_SCALE, ENV, MUTED, SCORE_SCALE, grid_figure
from aeris.dashboard.panels.grouping import group_choice, group_runs, label
from aeris.domain.reports import ZeroShotSuiteReport

ALL, ALL_LABEL = "all", "all tasks"  # the report's overall mean, shown after the families
DISTRIBUTIONS = ("score", "difference")


def _suite(run: RunData) -> ZeroShotSuiteReport | None:
    report = run.report(ZeroShotSuiteReport)
    return report if isinstance(report, ZeroShotSuiteReport) else None


def _protocol_choice(runs: Sequence[RunData]) -> Choice:
    protocols = tuple(
        dict.fromkeys(p for run in runs if (s := _suite(run)) for p in s.evidence_protocols)
    )
    default = (
        "dataset_rewards"
        if "dataset_rewards" in protocols
        else (protocols[0] if protocols else None)
    )
    return Choice("protocol", "Evidence", protocols, default)


def _fmt(value: float | None, percent: bool = False) -> str:
    if value is None:
        return "n/a"
    return f"{value:.0%}" if percent else f"{value:.2f}"


@dataclass(frozen=True)
class SuitePanel(Panel):
    """Shared requirement: every run (or the chosen run) has a suite report."""

    def missing(self, runs: Sequence[RunData]) -> str | None:
        absent = [run for run in runs if _suite(run) is None]
        if absent:
            return f"No zero-shot suite report. Run: aeris analysis zero-shot {absent[0].root}"
        return None


@dataclass(frozen=True)
class SuiteScores(SuitePanel):
    name = "suite_scores"
    title = "Zero-shot score by task family"
    scope = "runs"
    description = (
        "Mean normalized score (0 = random, 1 = optimal) per family; groups show ± std over seeds."
    )

    def options(self, runs: Sequence[RunData]) -> tuple[Option, ...]:
        return (_protocol_choice(runs), group_choice(runs))

    def render(self, runs: Sequence[RunData], options: Mapping[str, Any]) -> Output:
        protocol = options["protocol"]
        fig = go.Figure()
        lowest = 0.0
        for index, (group, members) in enumerate(group_runs(runs, options["group_by"]).items()):
            reports = [r for run in members if (r := _suite(run)) is not None]
            families = [f for f in dict.fromkeys(f for r in reports for f in r.mean_score)]
            families = [f for f in families if f != ALL] + [ALL]  # the overall mean last
            stats = [
                summarize(r.mean_score.get(f, {}).get(protocol) for r in reports) for f in families
            ]
            means = [s.mean if s else None for s in stats]
            lowest = min([lowest, *(m for m in means if m is not None)])
            fig.add_trace(
                go.Bar(
                    x=[ALL_LABEL if f == ALL else f for f in families],
                    y=means,
                    name=label(group, members),
                    marker_color=series_colour(index),
                    text=[_fmt(m) for m in means],
                    error_y={
                        "type": "data",
                        "array": [s.std if s else 0 for s in stats],
                        "visible": len(members) > 1,
                    },
                )
            )
        fig.update_layout(barmode="group")
        fig.update_yaxes(title="normalized score", range=[lowest - 0.05, 1.05])
        if fig.data:  # set the overall mean apart: its weight follows the family sizes
            fig.add_vline(x=len(fig.data[0].x) - 1.5, line={"color": MUTED, "dash": "dot"})
        return STYLE.apply(fig, f"{self.title} · {protocol}")


@dataclass(frozen=True)
class SuiteDistribution(SuitePanel):
    name = "suite_distribution"
    title = "Zero-shot scores per task"
    scope = "runs"
    description = (
        "Every task's normalized score, grouped by family: spread and outliers, not just means; "
        "or how much each task's score changes with the other evidence protocol."
    )

    def options(self, runs: Sequence[RunData]) -> tuple[Option, ...]:
        return (_protocol_choice(runs), Choice("show", "Show", DISTRIBUTIONS, "score"))

    def render(self, runs: Sequence[RunData], options: Mapping[str, Any]) -> Output:
        protocol = options["protocol"]
        difference = options["show"] == "difference"
        other = next(
            (
                p
                for run in runs
                if (r := _suite(run))
                for p in r.evidence_protocols
                if p != protocol
            ),
            None,
        )
        if difference and other is None:
            return Message("The suite report has only one evidence protocol.")
        fig = go.Figure()
        for index, run in enumerate(runs):
            report = _suite(run)
            assert report is not None
            scored = []
            for t in report.tasks:
                score = t.outcomes[protocol].normalized_score
                if difference and score is not None and other is not None:
                    base = t.outcomes[other].normalized_score
                    score = None if base is None else score - base
                if score is not None:
                    scored.append((t.family, score, t.task.task_id))
            fig.add_trace(
                go.Box(
                    x=[f for f, _, _ in scored],
                    y=[s for _, s, _ in scored],
                    text=[i for _, _, i in scored],
                    name=run.label,
                    marker_color=series_colour(index),
                    boxpoints="all",
                    jitter=0.4,
                    pointpos=0,
                    hovertemplate="%{text}: %{y:.3f}<extra></extra>",
                )
            )
        fig.update_layout(boxmode="group")
        if difference:
            fig.add_hline(y=0, line={"color": MUTED, "dash": "dot"})
            fig.update_yaxes(title=f"score difference ({protocol} − {other})")
            return STYLE.apply(fig, f"{self.title} · {protocol} − {other}")
        fig.update_yaxes(title="normalized score")
        return STYLE.apply(fig, f"{self.title} · {protocol}")


@dataclass(frozen=True)
class GoalScoresMap(SuitePanel):
    name = "goal_scores_map"
    title = "Zero-shot goal tasks"
    scope = "run"
    description = (
        "Each goal cell coloured and labelled by its task's score or goal success; "
        "× = no rewarded evidence."
    )

    def options(self, runs: Sequence[RunData]) -> tuple[Option, ...]:
        return (
            _protocol_choice(runs),
            Choice("metric", "Metric", ("normalized score", "goal success"), "normalized score"),
        )

    def render(self, runs: Sequence[RunData], options: Mapping[str, Any]) -> Output:
        report = _suite(runs[0])
        assert report is not None
        goals = [t for t in report.tasks if t.family == "goal"]
        if not goals:
            return Message("This suite has no goal tasks.")
        protocol, metric = options["protocol"], options["metric"]

        def value(result) -> float | None:
            outcome = result.outcomes[protocol]
            return (
                outcome.normalized_score if metric == "normalized score" else outcome.goal_success
            )

        cells: dict[tuple[int, ...], float] = {
            ENV.goal_from_task(t.task): v for t in goals if (v := value(t)) is not None
        }
        fig = grid_figure(
            cells,
            colorbar_title=metric,
            **(
                {"zmin": -1.0, "zmax": 1.0, "colorscale": SCORE_SCALE, "text_format": "%{z:.2f}"}
                if metric == "normalized score"
                else {"colorscale": COUNT_SCALE, "text_format": "%{z:.0%}"}
            ),
        )
        invisible = [
            ENV.goal_from_task(t.task) for t in goals if t.outcomes[protocol].rewarded_samples == 0
        ]
        if invisible:
            fig.add_trace(
                go.Scatter(
                    x=[g[1] for g in invisible],
                    y=[g[0] for g in invisible],
                    mode="text",
                    text=["×"] * len(invisible),
                    name="no rewarded evidence",
                    hoverinfo="skip",
                )
            )
        mean = sum(cells.values()) / len(cells) if cells else None
        return STYLE.apply(fig, f"{self.title} · {protocol} · mean {metric} {_fmt(mean)}")


@dataclass(frozen=True)
class SuiteTasks(SuitePanel):
    name = "suite_tasks"
    title = "Suite tasks"
    scope = "run"
    description = "Every task's results, one row per task."

    def options(self, runs: Sequence[RunData]) -> tuple[Option, ...]:
        report = _suite(runs[0])
        families = ("all", *dict.fromkeys(t.family for t in report.tasks)) if report else ("all",)
        return (
            Choice("family", "Family", families, "all"),
            _protocol_choice(runs),
            Choice("order", "Order", ("lowest score first", "suite order"), "lowest score first"),
        )

    def render(self, runs: Sequence[RunData], options: Mapping[str, Any]) -> Output:
        report = _suite(runs[0])
        assert report is not None
        main = options["protocol"]
        others = [p for p in report.evidence_protocols if p != main]
        rows = []
        for result in report.tasks:
            if options["family"] not in ("all", result.family):
                continue
            outcomes = result.outcomes
            row: dict[str, Any] = {"task": result.task.task_id, "family": result.family}
            for protocol in (main, *others):
                row[f"score · {protocol}"] = _round(outcomes[protocol].normalized_score)
            for protocol in others:
                row[f"gap · {main} − {protocol}"] = _round(
                    _minus(outcomes[main].normalized_score, outcomes[protocol].normalized_score)
                )
            for protocol in (main, *others):
                if outcomes[protocol].goal_success is not None:
                    row[f"success · {protocol}"] = _round(outcomes[protocol].goal_success)
            row["rewarded samples"] = (
                outcomes["dataset_rewards"].rewarded_samples
                if "dataset_rewards" in outcomes
                else None
            )
            rows.append(row)
        if options["order"] == "lowest score first":
            key = f"score · {main}"
            rows.sort(key=lambda row: (row[key] is None, row[key]))
        return Table(rows, caption=f"{report.suite}: {report.suite_description}")


def _round(value: float | None) -> float | None:
    """Three decimals: enough to read, and the column still sorts as numbers."""
    return None if value is None else round(value, 3)


def _minus(a: float | None, b: float | None) -> float | None:
    return None if a is None or b is None else a - b
