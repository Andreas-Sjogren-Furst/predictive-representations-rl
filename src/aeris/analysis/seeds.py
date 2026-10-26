"""Results across seeds: each run's key metrics, grouped and summarized.

A group is every run that shares a recorded value (by default the experiment
name, which a seed sweep gives all its seeds). Metrics come from the typed
records a run saved, so nothing is recomputed.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from aeris.agents.fb.reports import FBCheckpointReport
from aeris.domain.reports import AnalysisReport, TabularPolicyReport, ZeroShotSuiteReport
from aeris.domain.results import EvaluationRecord, RunManifest

# Report types whose headline numbers ``key_metrics`` reads.
REPORT_TYPES: tuple[type[AnalysisReport], ...] = (
    TabularPolicyReport,
    FBCheckpointReport,
    ZeroShotSuiteReport,
)


@dataclass(frozen=True, slots=True)
class SeedStatistic:
    mean: float
    std: float  # sample standard deviation across seeds (0 for a single seed)
    n: int  # seeds that reported the metric

    def __str__(self) -> str:
        return f"{self.mean:.3f} ± {self.std:.3f} (n={self.n})"


def summarize(values: Iterable[float | None]) -> SeedStatistic | None:
    """Mean and sample standard deviation of the values that exist."""
    defined = np.array([v for v in values if v is not None], dtype=np.float64)
    if len(defined) == 0:
        return None
    std = float(defined.std(ddof=1)) if len(defined) > 1 else 0.0
    return SeedStatistic(mean=float(defined.mean()), std=std, n=len(defined))


def key_metrics(
    evaluations: Sequence[EvaluationRecord], reports: Iterable[AnalysisReport]
) -> dict[str, float | None]:
    """The headline numbers of one run, by name; absent analyses give no entries."""
    metrics: dict[str, float | None] = {}
    final = evaluations[-1] if evaluations else None
    if final is not None:
        for metric in final.metrics:
            metrics[f"final evaluation · {metric.name}"] = metric.value
    for report in reports:
        if isinstance(report, TabularPolicyReport) and "final" in report.snapshots:
            snapshot = report.snapshots["final"]
            metrics["final policy · optimal actions"] = snapshot.optimal_action_fraction
            metrics["final policy · all-start success"] = snapshot.rollout_success_fraction
        elif isinstance(report, FBCheckpointReport):
            metrics["FB · occupancy correlation"] = report.exact_occupancy_correlation
            metrics["FB · policy-improvement agreement"] = (
                report.goal_action_agreement_with_exact_policy_improvement
            )
        elif isinstance(report, ZeroShotSuiteReport):
            for family, means in report.mean_score.items():
                for protocol, value in means.items():
                    metrics[f"zero-shot · {family} · {protocol}"] = value
    return metrics


def group_key(manifest: RunManifest, by: str) -> str:
    """The group a run belongs to: its recorded value at ``by``, e.g. ``name``."""
    return str(manifest.setting(by))


@dataclass(frozen=True, slots=True)
class GroupSummary:
    group: str
    runs: tuple[str, ...]
    seeds: tuple[int, ...]
    metrics: Mapping[str, SeedStatistic]


def summarize_groups(
    runs: Sequence[tuple[str, RunManifest, Mapping[str, float | None]]], *, by: str = "name"
) -> list[GroupSummary]:
    """``runs`` holds (label, manifest, key metrics) per run; groups keep first-seen order."""
    groups: dict[str, list[tuple[str, RunManifest, Mapping[str, float | None]]]] = {}
    for run in runs:
        groups.setdefault(group_key(run[1], by), []).append(run)
    summaries = []
    for group, members in groups.items():
        names = list(dict.fromkeys(name for _, _, metrics in members for name in metrics))
        statistics = {
            name: stat
            for name in names
            if (stat := summarize(metrics.get(name) for _, _, metrics in members)) is not None
        }
        summaries.append(
            GroupSummary(
                group=group,
                runs=tuple(label for label, _, _ in members),
                seeds=tuple(manifest.experiment.seed for _, manifest, _ in members),
                metrics=statistics,
            )
        )
    return summaries
