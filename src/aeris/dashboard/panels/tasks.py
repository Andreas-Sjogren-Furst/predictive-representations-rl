"""Zero-shot task-suite panels, drawn from a saved ``zero_shot_suite_v1`` report.

Create the report with ``aeris analysis zero-shot RUN``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import plotly.graph_objects as go

from aeris.dashboard.data import RunData
from aeris.dashboard.panels.base import STYLE, Choice, Message, Option, Output, Panel, Table
from aeris.dashboard.panels.grid import ARROWS, ENV, add_goal, grid_figure
from aeris.domain.reports import ZeroShotSuiteReport


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
    description = "Mean normalized score (0 = random policy, 1 = optimal) per task family and run."

    def options(self, runs: Sequence[RunData]) -> tuple[Option, ...]:
        return (_protocol_choice(runs),)

    def render(self, runs: Sequence[RunData], options: Mapping[str, Any]) -> Output:
        protocol = options["protocol"]
        fig = go.Figure()
        lowest = 0.0
        for run in runs:
            report = _suite(run)
            assert report is not None
            families = list(report.mean_score)
            values = [report.mean_score[f].get(protocol) for f in families]
            lowest = min([lowest, *(v for v in values if v is not None)])
            fig.add_trace(
                go.Bar(x=families, y=values, name=run.label, text=[_fmt(v) for v in values])
            )
        fig.update_layout(barmode="group")
        fig.update_yaxes(title="normalized score", range=[lowest - 0.05, 1.05])
        return STYLE.apply(fig, f"{self.title} · {protocol}")


@dataclass(frozen=True)
class SuiteDistribution(SuitePanel):
    name = "suite_distribution"
    title = "Zero-shot scores per task"
    scope = "runs"
    description = (
        "Every task's normalized score, grouped by family: spread and outliers, not just means."
    )

    def options(self, runs: Sequence[RunData]) -> tuple[Option, ...]:
        return (_protocol_choice(runs),)

    def render(self, runs: Sequence[RunData], options: Mapping[str, Any]) -> Output:
        protocol = options["protocol"]
        fig = go.Figure()
        for run in runs:
            report = _suite(run)
            assert report is not None
            scored = [
                (t.family, t.outcomes[protocol].normalized_score, t.task.task_id)
                for t in report.tasks
            ]
            scored = [(f, s, i) for f, s, i in scored if s is not None]
            fig.add_trace(
                go.Box(
                    x=[f for f, _, _ in scored],
                    y=[s for _, s, _ in scored],
                    text=[i for _, _, i in scored],
                    name=run.label,
                    boxpoints="all",
                    jitter=0.4,
                    pointpos=0,
                    hovertemplate="%{text}: %{y:.2f}<extra></extra>",
                )
            )
        fig.update_layout(boxmode="group")
        fig.update_yaxes(title="normalized score")
        return STYLE.apply(fig, f"{self.title} · {protocol}")


@dataclass(frozen=True)
class TaskPolicy(SuitePanel):
    name = "task_policy"
    title = "Task policy"
    scope = "run"
    description = "One task's reward in every cell with the action the agent inferred for it."

    def options(self, runs: Sequence[RunData]) -> tuple[Option, ...]:
        report = _suite(runs[0])
        ids = tuple(t.task.task_id for t in report.tasks) if report else ()
        default = next((i for i in ids if i.startswith("room_")), ids[0] if ids else None)
        return (Choice("task", "Task", ids, default), _protocol_choice(runs))

    def render(self, runs: Sequence[RunData], options: Mapping[str, Any]) -> Output:
        report = _suite(runs[0])
        assert report is not None
        result = report.result(options["task"])
        outcome = result.outcomes[options["protocol"]]
        rewards = ENV.state_rewards(result.task)
        values = {tuple(s): float(r) for s, r in zip(report.states, rewards, strict=True)}
        low, high = float(rewards.min()), float(rewards.max())
        if low == high:
            low, high = low - 1, high + 1
        fig = grid_figure(values, zmin=low, zmax=high, colorscale="RdBu", colorbar_title="reward")
        goal = ENV.goal_from_task(result.task) if result.family in ("goal", "distance") else None
        cells = [
            (s, a) for s, a in zip(report.states, outcome.actions, strict=True) if tuple(s) != goal
        ]
        fig.add_trace(
            go.Scatter(
                x=[s[1] for s, _ in cells],
                y=[s[0] for s, _ in cells],
                mode="text",
                text=[ARROWS[a] for _, a in cells],
                textfont={"size": 15, "color": "#111111"},
                hoverinfo="skip",
                showlegend=False,
            )
        )
        if goal is not None:
            add_goal(fig, goal)
        parts = [f"score {_fmt(outcome.normalized_score)}"]
        if outcome.goal_success is not None:
            parts.append(f"goal reached from {_fmt(outcome.goal_success, percent=True)} of starts")
        if outcome.rewarded_samples == 0:
            parts.append("no rewarded evidence")
        return STYLE.apply(
            fig, f"{result.task.task_id} · {options['protocol']} · " + " · ".join(parts)
        )


@dataclass(frozen=True)
class GoalScoresMap(SuitePanel):
    name = "goal_scores_map"
    title = "Zero-shot goal tasks"
    scope = "run"
    description = (
        "Each goal cell coloured by its task's score or goal success; × = no rewarded evidence."
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
        fig = grid_figure(cells, colorbar_title=metric)
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
        return (Choice("family", "Family", families, "all"),)

    def render(self, runs: Sequence[RunData], options: Mapping[str, Any]) -> Output:
        report = _suite(runs[0])
        assert report is not None
        rows = []
        for result in report.tasks:
            if options["family"] not in ("all", result.family):
                continue
            row: dict[str, Any] = {"task": result.task.task_id, "family": result.family}
            for protocol, outcome in result.outcomes.items():
                row[f"score · {protocol}"] = outcome.normalized_score
                if outcome.goal_success is not None:
                    row[f"success · {protocol}"] = outcome.goal_success
            row["rewarded samples"] = (
                result.outcomes["dataset_rewards"].rewarded_samples
                if "dataset_rewards" in result.outcomes
                else None
            )
            rows.append(row)
        return Table(rows, caption=f"{report.suite}: {report.suite_description}")
