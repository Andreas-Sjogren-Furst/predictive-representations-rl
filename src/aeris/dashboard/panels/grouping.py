"""Grouping runs by a recorded value (e.g. seeds of one experiment) for aggregated panels."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from aeris.dashboard.data import RunData
from aeris.dashboard.panels.base import Choice

NO_GROUPING = "run"
_COMPONENTS = ("agent", "dataset", "environment", "training", "evaluation")


def group_choice(runs: Sequence[RunData]) -> Choice:
    """'run' (no grouping), the experiment name, and every setting that differs between runs."""
    candidates = ["name"]
    for component in _COMPONENTS:
        keys = dict.fromkeys(key for run in runs for key in (_settings(run, component) or {}))
        candidates += [f"{component}.{key}" for key in keys]
    varying = [path for path in candidates if len({str(run.setting(path)) for run in runs}) > 1]
    choices = (NO_GROUPING, *dict.fromkeys(["name", *varying]))
    return Choice("group_by", "Group runs by", choices, NO_GROUPING)


def _settings(run: RunData, component: str):
    experiment = run.manifest.experiment
    spec = {
        "agent": experiment.agent,
        "dataset": experiment.dataset,
        "environment": experiment.environment,
        "training": experiment.training.strategy,
        "evaluation": experiment.evaluation.protocol if experiment.evaluation else None,
    }[component]
    return None if spec is None else spec.settings


def group_runs(runs: Sequence[RunData], by: str) -> dict[str, list[RunData]]:
    """Runs per group label, in first-seen order; ``"run"`` keeps every run on its own."""
    groups: dict[str, list[RunData]] = {}
    for run in runs:
        label = run.label if by == NO_GROUPING else f"{by}={run.setting(by)}"
        groups.setdefault(label, []).append(run)
    return groups


def label(group: str, members: Sequence[RunData]) -> str:
    return group if len(members) == 1 else f"{group} (n={len(members)})"


def mean_and_std(
    series: Sequence[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean and sample std across runs at the steps every run logged."""
    common = series[0][0]
    for steps, _ in series[1:]:
        common = np.intersect1d(common, steps)
    stacked = np.stack([values[np.isin(steps, common)] for steps, values in series])
    std = stacked.std(axis=0, ddof=1) if len(series) > 1 else np.zeros(len(common))
    return common, stacked.mean(axis=0), std
