"""Tables: run overview, any saved report, and a comparison across runs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from aeris.agents.fb.reports import FBCheckpointReport
from aeris.analysis.seeds import (
    Interval,
    SeedStatistic,
    difference,
    key_metrics,
    summarize,
)
from aeris.dashboard.data import RunData
from aeris.dashboard.panels.base import Choice, MultiChoice, Option, Output, Panel, Table
from aeris.dashboard.panels.grouping import group_choice, group_runs, label
from aeris.domain.reports import AnalysisReport, TabularPolicyReport, ZeroShotSuiteReport


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


def display(value: Any) -> str:
    """A table cell: floats rounded to what can be read, everything else as text."""
    if isinstance(value, float):
        if 0.01 <= abs(value) < 1e6:
            return f"{value:.3f}".rstrip("0").rstrip(".")
        return f"{value:.3g}"
    return str(value)


# Settings by section, in reading order; provenance is hidden unless asked for.
SECTIONS = (
    "run",
    "experiment",
    "dataset",
    "agent",
    "training",
    "environment",
    "evaluation",
    "provenance",
)


def _section(key: str) -> tuple[str, str]:
    """A dotted manifest key as (section, short name): ``experiment.agent.settings.gamma``
    is ("agent", "gamma")."""
    parts = [p for p in key.split(".") if p != "settings"]
    if parts[0] == "experiment" and len(parts) > 2:
        return parts[1], ".".join(parts[2:])
    if parts[0] in ("experiment", "provenance") and len(parts) > 1:
        return parts[0], ".".join(parts[1:])
    return "run", key


def _flag(section: str, name: str, value: Any) -> str:
    """Warn where a value means the run may not be what it claims to be."""
    if (name == "git_dirty" and value is True) or (name == "error" and value is not None):
        return f"⚠ {value}"
    if name == "status" and value != "completed":
        return f"⚠ {value}"
    return display(value)


@dataclass(frozen=True)
class RunOverview(Panel):
    name = "run_overview"
    title = "Run settings"
    scope = "run"
    description = "Status, provenance and every recorded setting of the run, by section."

    def options(self, runs: Sequence[RunData]) -> tuple[Option, ...]:
        shown = tuple(s for s in SECTIONS if s != "provenance")
        return (MultiChoice("sections", "Sections", SECTIONS, shown),)

    def render(self, runs: Sequence[RunData], options: Mapping[str, Any]) -> Output:
        run = runs[0]
        flat = {"directory": str(run.root), **_scalars(run.manifest.to_dict())}
        rows = [
            {"section": section, "setting": name, "value": _flag(section, name, value)}
            for key, value in flat.items()
            for section, name in [_section(key)]
            if section in options["sections"]
        ]
        rows.sort(key=lambda row: SECTIONS.index(row["section"]))  # stable within a section
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
        rows = [{"field": k, "value": display(v)} for k, v in _scalars(report.to_dict()).items()]
        return Table(rows, caption=f"{options['report']} ({type(report).ARTIFACT_NAME})")


def _final_success(run: RunData) -> float | None:
    for record in reversed(run.evaluations):
        for metric in record.metrics:
            if metric.name == "success_rate":
                return metric.value
    return None


def _episodes(runs: Sequence[RunData]) -> str:
    counts = sorted({run.evaluations[-1].episode_count for run in runs if run.evaluations})
    return "/".join(str(c) for c in counts) or "?"


# The analysis that saves each report, for "missing" notes.
ANALYSES: tuple[tuple[type[AnalysisReport], str], ...] = (
    (TabularPolicyReport, "four-rooms"),
    (FBCheckpointReport, "fb-checkpoint"),
    (ZeroShotSuiteReport, "zero-shot"),
)


def missing_analyses(runs: Sequence[RunData]) -> str:
    """Which analyses some selected runs lack, with the command for the first of them."""
    notes = []
    for report_type, command in ANALYSES:
        absent = [run for run in runs if not isinstance(run.report(report_type), report_type)]
        if absent:
            notes.append(
                f"{command} for {len(absent)} of {len(runs)} runs "
                f"(aeris analysis {command} {absent[0].root})"
            )
    return "Missing analyses: " + "; ".join(notes) if notes else ""


def metric_order(name: str, episodes: str | None = None) -> tuple[int, str, str]:
    """A ``key_metrics`` name as (rank, section, label), exact results first and the
    sampled evaluation last: ``zero-shot · goal · all_states`` is (…, "zero-shot ·
    all_states", "goal")."""
    parts = name.split(" · ")
    if parts[0] == "final policy":
        return (0 if parts[1] == "all-start success" else 1), "training goal · exact", parts[1]
    if parts[0] == "FB":
        return 10, "FB checkpoint", parts[1]
    if parts[0] == "zero-shot" and len(parts) == 3:
        family, protocol = parts[1], parts[2]
        rank = 20 + (0 if protocol == "dataset_rewards" else 1) * 10 + (0 if family == "all" else 1)
        return rank, f"zero-shot · {protocol}", "all tasks" if family == "all" else family
    if parts[0] == "final evaluation":
        section = "sampled evaluation" + (f" · {episodes} episodes" if episodes else "")
        return 90, section, " · ".join(parts[1:])
    return 50, "other", name


def run_metrics(runs: Sequence[RunData]) -> dict[str, list[float | None]]:
    """Each headline number across ``runs`` (None where a run lacks it), in reading order."""
    per_run = [key_metrics(run.evaluations, run.reports.values()) for run in runs]
    names = sorted(
        dict.fromkeys(n for metrics in per_run for n in metrics),
        key=lambda n: metric_order(n)[0],  # stable: the reports' own order within a rank
    )
    return {name: [metrics.get(name) for metrics in per_run] for name in names}


def signed(value: float) -> str:
    return f"{value:+.3f}".replace("-", "−")


def interval_text(interval: Interval | None) -> str:
    if interval is None:
        return ""
    mark = " ✓" if interval.excludes_zero else ""
    return f"{signed(interval.value)} [{signed(interval.low)}, {signed(interval.high)}]{mark}"


@dataclass(frozen=True)
class RunComparison(Panel):
    name = "run_comparison"
    title = "Run comparison"
    scope = "runs"
    description = "Key settings and results of every selected run, grouped by experiment."

    def render(self, runs: Sequence[RunData], options: Mapping[str, Any]) -> Output:
        names = list(dict.fromkeys(run.manifest.experiment.name for run in runs))
        ordered = sorted(
            runs,
            key=lambda run: (
                names.index(run.manifest.experiment.name),
                run.manifest.experiment.seed,
            ),
        )
        rows = []
        for run in ordered:
            tabular = run.report(TabularPolicyReport)
            fb = run.report(FBCheckpointReport)
            suite = run.report(ZeroShotSuiteReport)
            final = (
                tabular.snapshots.get("final") if isinstance(tabular, TabularPolicyReport) else None
            )
            rows.append(
                {
                    "experiment": run.manifest.experiment.name,
                    "seed": run.manifest.experiment.seed,
                    "run": run.label,
                    "status": run.manifest.status,
                    "dataset": run.setting("dataset.origin"),
                    "updates": _strategy(run, "updates"),
                    "hidden": run.setting("agent.hidden_dim"),
                    "d": run.setting("agent.representation_dim"),
                    "all-start success": _round(final.rollout_success_fraction if final else None),
                    "optimal actions": _round(final.optimal_action_fraction if final else None),
                    "occupancy corr": _round(fb.exact_occupancy_correlation)
                    if isinstance(fb, FBCheckpointReport)
                    else None,
                    **(
                        {f"zero-shot {p}": _round(v) for p, v in suite.mean_score["all"].items()}
                        if isinstance(suite, ZeroShotSuiteReport)
                        else {}
                    ),
                    "sampled success": _round(_final_success(run)),
                }
            )
        caption = " · ".join(
            part
            for part in (
                f"{self.title}: training goal exact over every start; sampled success from "
                f"the last evaluation ({_episodes(runs)} episodes)",
                missing_analyses(runs),
            )
            if part
        )
        return Table(rows, caption=caption)


def _round(value: float | None) -> float | None:
    """Three decimals: enough to read, and the column still sorts as numbers."""
    return None if value is None else round(value, 3)


def _strategy(run: RunData, key: str) -> Any:
    strategy = run.manifest.experiment.training.strategy
    return None if strategy is None else strategy.settings.get(key)


@dataclass(frozen=True)
class GroupSummaryTable(Panel):
    name = "group_summary"
    title = "Group summary"
    scope = "runs"
    description = (
        "Mean ± std across the runs (seeds) of each group for every headline number, and each "
        "group's difference from the first with a 95% interval (Welch); ✓ where it excludes 0."
    )

    def options(self, runs: Sequence[RunData]) -> tuple[Option, ...]:
        choice = group_choice(runs)
        default = "name" if "name" in choice.choices else choice.default
        return (Choice(choice.key, choice.label, choice.choices, default),)

    def render(self, runs: Sequence[RunData], options: Mapping[str, Any]) -> Output:
        groups = group_runs(runs, options["group_by"])
        values = {group: run_metrics(members) for group, members in groups.items()}
        names = sorted(
            dict.fromkeys(n for metrics in values.values() for n in metrics),
            key=lambda n: metric_order(n)[0],
        )
        first = next(iter(groups))
        episodes = _episodes(runs)
        rows = []
        for name in names:
            _, section, metric = metric_order(name, episodes)
            row: dict[str, Any] = {"section": section, "metric": metric}
            for group, members in groups.items():
                stat = summarize(values[group].get(name, []))
                row[label(group, members)] = _stat(stat, len(members))
            for group in list(groups)[1:]:
                row[f"{group} − {first}"] = interval_text(
                    difference(values[first].get(name, []), values[group].get(name, []))
                )
            rows.append(row)
        caption = " · ".join(
            part
            for part in (
                f"{self.title} · grouped by {options['group_by']} · difference: mean with "
                "95% interval (Welch), ✓ where it excludes 0",
                missing_analyses(runs),
            )
            if part
        )
        return Table(rows, caption=caption)


def _stat(stat: SeedStatistic | None, size: int) -> str:
    if stat is None:
        return "no report"
    if stat.n == 1:
        return f"{stat.mean:.3f}"
    count = "" if stat.n == size else f" (n={stat.n})"
    return f"{stat.mean:.3f} ± {stat.std:.3f}{count}"
