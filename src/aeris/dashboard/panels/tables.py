"""Tables: run overview, any saved report, and a comparison across runs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from aeris.agents.fb.reports import FBCheckpointReport
from aeris.dashboard.data import RunData
from aeris.dashboard.panels.base import Choice, Option, Output, Panel, Table
from aeris.domain.reports import TabularPolicyReport, ZeroShotReport


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
            zero_shot = run.report(ZeroShotReport)
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
                        {f"zero-shot {k}": v for k, v in zero_shot.mean_success_rate.items()}
                        if isinstance(zero_shot, ZeroShotReport)
                        else {}
                    ),
                }
            )
        return Table(rows, caption=self.title)


def _strategy(run: RunData, key: str) -> Any:
    strategy = run.manifest.experiment.training.strategy
    return None if strategy is None else strategy.settings.get(key)
