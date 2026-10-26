"""Tables: run overview, any saved report, and a comparison across runs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from aeris.agents.fb.reports import FBCheckpointReport
from aeris.analysis.seeds import SeedStatistic, key_metrics, summarize
from aeris.dashboard.data import RunData
from aeris.dashboard.panels.base import Choice, Option, Output, Panel, Table
from aeris.dashboard.panels.grouping import group_choice, group_runs, label
from aeris.domain.reports import TabularPolicyReport, ZeroShotSuiteReport


def _scalars(data: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten nested JSON data to dotted keys, skipping long per-state lists."""
    flat: dict[str, Any] = {}
    for key, value in data.items():
        name = f"{prefix}{key}"
        if isinstance(value, Mapping):
            flat.update(_scalars(value, f"{name}."))
        elif isinstance(value, list):
            if len(value) <= 4 and all(not isinstance(v, (list, dict)) for v in value):
                flat[name] = str(value)
        else:
            flat[name] = value
    return flat


@dataclass(frozen=True)
class RunOverview(Panel):
    name = "run_overview"
    title = "Run overview"
    scope = "run"
    description = "Status, provenance and every recorded setting of the run."

    def render(self, runs: Sequence[RunData], options: Mapping[str, Any]) -> Output:
        run = runs[0]
        manifest = run.manifest.to_dict()
        rows = [{"setting": "directory", "value": str(run.root)}]
        rows += [{"setting": k, "value": str(v)} for k, v in _scalars(manifest).items()]
        return Table(rows, caption=self.title)


@dataclass(frozen=True)
class ReportView(Panel):
    name = "report"
    title = "Analysis report"
    scope = "run"
    description = "The numbers of one saved analysis report."

    def options(self, runs: Sequence[RunData]) -> tuple[Option, ...]:
        names = tuple(runs[0].reports)
        return (Choice("report", "Report", names, names[0] if names else None),)

    def missing(self, runs: Sequence[RunData]) -> str | None:
        return (
            None
            if runs[0].reports
            else f"No analysis reports. Run e.g.: aeris analysis four-rooms {runs[0].root}"
        )

    def render(self, runs: Sequence[RunData], options: Mapping[str, Any]) -> Output:
        report = runs[0].reports[options["report"]]
        rows = [{"field": k, "value": str(v)} for k, v in _scalars(report.to_dict()).items()]
        return Table(rows, caption=f"{options['report']} ({type(report).ARTIFACT_NAME})")


def _final_success(run: RunData) -> float | None:
    for record in reversed(run.evaluations):
        for metric in record.metrics:
            if metric.name == "success_rate":
                return metric.value
    return None


@dataclass(frozen=True)
class RunComparison(Panel):
    name = "run_comparison"
    title = "Run comparison"
    scope = "runs"
    description = "Key settings and results of every selected run side by side."

    def render(self, runs: Sequence[RunData], options: Mapping[str, Any]) -> Output:
        rows = []
        for run in runs:
            tabular = run.report(TabularPolicyReport)
            fb = run.report(FBCheckpointReport)
            suite = run.report(ZeroShotSuiteReport)
            final = (
                tabular.snapshots.get("final") if isinstance(tabular, TabularPolicyReport) else None
            )
            rows.append(
                {
                    "run": run.label,
                    "status": run.manifest.status,
                    "dataset": run.setting("dataset.origin"),
                    "updates": _strategy(run, "updates"),
                    "hidden": run.setting("agent.hidden_dim"),
                    "d": run.setting("agent.representation_dim"),
                    "seed": run.manifest.experiment.seed,
                    "final success": _final_success(run),
                    "optimal actions": final.optimal_action_fraction if final else None,
                    "all-start success": final.rollout_success_fraction if final else None,
                    "occupancy corr": fb.exact_occupancy_correlation
                    if isinstance(fb, FBCheckpointReport)
                    else None,
                    **(
                        {f"zero-shot {p}": v for p, v in suite.mean_score["all"].items()}
                        if isinstance(suite, ZeroShotSuiteReport)
                        else {}
                    ),
                }
            )
        return Table(rows, caption=self.title)


def _strategy(run: RunData, key: str) -> Any:
    strategy = run.manifest.experiment.training.strategy
    return None if strategy is None else strategy.settings.get(key)


@dataclass(frozen=True)
class GroupSummaryTable(Panel):
    name = "group_summary"
    title = "Group summary"
    scope = "runs"
    description = "Mean ± std across the runs (seeds) of each group, for every headline number."

    def options(self, runs: Sequence[RunData]) -> tuple[Option, ...]:
        choice = group_choice(runs)
        default = "name" if "name" in choice.choices else choice.default
        return (Choice(choice.key, choice.label, choice.choices, default),)

    def render(self, runs: Sequence[RunData], options: Mapping[str, Any]) -> Output:
        groups = group_runs(runs, options["group_by"])
        columns = {
            label(group, members): {
                name: stat
                for name in dict.fromkeys(
                    n for run in members for n in key_metrics(run.evaluations, run.reports.values())
                )
                if (
                    stat := summarize(
                        key_metrics(run.evaluations, run.reports.values()).get(name)
                        for run in members
                    )
                )
                is not None
            }
            for group, members in groups.items()
        }
        metrics = list(dict.fromkeys(name for stats in columns.values() for name in stats))
        rows = [
            {"metric": name, **{group: _stat(stats.get(name)) for group, stats in columns.items()}}
            for name in metrics
        ]
        return Table(rows, caption=f"{self.title} · grouped by {options['group_by']}")


def _stat(stat: SeedStatistic | None) -> str:
    if stat is None:
        return ""
    return f"{stat.mean:.3f}" if stat.n == 1 else f"{stat.mean:.3f} ± {stat.std:.3f} (n={stat.n})"
