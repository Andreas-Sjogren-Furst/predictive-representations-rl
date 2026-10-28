"""Results across seeds: each run's key metrics, grouped and summarized.

A group is every run that shares a recorded value (by default the experiment
name, which a seed sweep gives all its seeds). Metrics come from the typed
records a run saved, so nothing is recomputed.
"""

from __future__ import annotations

import math
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


def t_quantile(probability: float, df: float) -> float:
    """The ``probability`` quantile of Student's t with ``df`` (possibly fractional) degrees
    of freedom, for ``probability`` above 0.5: bisection on the integrated density."""
    if not 0.5 < probability < 1 or df <= 0:
        raise ValueError("needs 0.5 < probability < 1 and df > 0")
    scale = math.exp(math.lgamma((df + 1) / 2) - math.lgamma(df / 2)) / math.sqrt(df * math.pi)

    def cdf(t: float) -> float:  # Simpson's rule from 0 to t; the density is smooth
        x = np.linspace(0.0, t, 2001)
        y = scale * (1 + x**2 / df) ** (-(df + 1) / 2)
        h = t / 2000
        return 0.5 + h / 3 * float(y[0] + y[-1] + 4 * y[1:-1:2].sum() + 2 * y[2:-1:2].sum())

    low, high = 0.0, 1.0
    while cdf(high) < probability:
        low, high = high, 2 * high
    for _ in range(60):
        middle = (low + high) / 2
        low, high = (middle, high) if cdf(middle) < probability else (low, middle)
    return (low + high) / 2


@dataclass(frozen=True, slots=True)
class Interval:
    """An estimate with a confidence interval."""

    value: float
    low: float
    high: float

    @property
    def excludes_zero(self) -> bool:
        return self.low > 0 or self.high < 0


def mean_interval(values: Iterable[float | None], level: float = 0.95) -> Interval | None:
    """The mean of the values that exist, with a t interval; None for fewer than two."""
    defined = np.array([v for v in values if v is not None], dtype=np.float64)
    if len(defined) < 2:
        return None
    mean = float(defined.mean())
    half = t_quantile((1 + level) / 2, len(defined) - 1) * defined.std(ddof=1) / len(defined) ** 0.5
    return Interval(mean, mean - float(half), mean + float(half))


def difference(
    first: Iterable[float | None], second: Iterable[float | None], level: float = 0.95
) -> Interval | None:
    """Mean of ``second`` minus mean of ``first``, with a Welch t interval (unequal
    variances); None unless both have at least two values."""
    a = np.array([v for v in first if v is not None], dtype=np.float64)
    b = np.array([v for v in second if v is not None], dtype=np.float64)
    if len(a) < 2 or len(b) < 2:
        return None
    value = float(b.mean() - a.mean())
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    if va + vb == 0:
        return Interval(value, value, value)
    df = (va + vb) ** 2 / (va**2 / (len(a) - 1) + vb**2 / (len(b) - 1))
    half = t_quantile((1 + level) / 2, float(df)) * math.sqrt(va + vb)
    return Interval(value, value - half, value + half)


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
