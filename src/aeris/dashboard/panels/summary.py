"""The headline numbers of one run, as a row of stat tiles."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from aeris.agents.fb.reports import FBCheckpointReport
from aeris.dashboard.data import RunData
from aeris.dashboard.panels.base import Output, Panel, Stat, Stats
from aeris.domain.reports import TabularPolicyReport, ZeroShotSuiteReport


def _percent(value: float) -> str:
    return f"{value:.0%}"


def _join(*parts: str) -> str:
    return " · ".join(part for part in parts if part)


def _points(final: float, initial: float) -> str:
    return f"{(final - initial) * 100:+.0f} pts vs initial"


@dataclass(frozen=True)
class RunSummary(Panel):
    name = "run_summary"
    title = "Run summary"
    scope = "run"
    description = (
        "Status, setup and the exact headline results: training goal, representation "
        "quality and zero-shot scores."
    )

    def render(self, runs: Sequence[RunData], options: Mapping[str, Any]) -> Output:
        run = runs[0]
        stats = [self._status(run), self._setup(run)]
        absent = []
        tabular = run.report(TabularPolicyReport)
        if isinstance(tabular, TabularPolicyReport) and "final" in tabular.snapshots:
            stats += self._training_goal(tabular)
        else:
            absent.append(f"aeris analysis four-rooms {run.root}")
        fb = run.report(FBCheckpointReport)
        if isinstance(fb, FBCheckpointReport) and fb.exact_occupancy_correlation is not None:
            stats.append(
                Stat(
                    "Occupancy correlation",
                    f"{fb.exact_occupancy_correlation:.3f}",
                    "F·B against the exact successor measure",
                )
            )
        else:
            absent.append(f"aeris analysis fb-checkpoint {run.root}")
        suite = run.report(ZeroShotSuiteReport)
        if isinstance(suite, ZeroShotSuiteReport):
            stats += self._zero_shot(suite)
        else:
            absent.append(f"aeris analysis zero-shot {run.root}")
        caption = "Missing analyses: " + " · ".join(absent) if absent else ""
        return Stats(stats, caption=caption)

    @staticmethod
    def _status(run: RunData) -> Stat:
        manifest = run.manifest
        return Stat(
            "Status",
            manifest.status,
            _join(
                f"{manifest.experiment.name} · seed {manifest.experiment.seed}",
                "⚠ uncommitted code" if manifest.provenance.get("git_dirty") else "",
                f"⚠ {manifest.error}" if manifest.error else "",
            ),
        )

    @staticmethod
    def _setup(run: RunData) -> Stat:
        strategy = run.manifest.experiment.training.strategy
        updates = strategy.settings.get("updates") if strategy is not None else None
        value = f"{updates:,} updates" if isinstance(updates, int) else "training"
        parts = []
        if (origin := run.setting("dataset.origin")) is not None:
            count = run.setting("dataset.count")
            parts.append(f"{origin} data" + (f", {count:,}" if isinstance(count, int) else ""))
        if (dim := run.setting("agent.representation_dim")) is not None:
            parts.append(f"d = {dim}")
        return Stat("Setup", value, _join(*parts))

    @staticmethod
    def _training_goal(report: TabularPolicyReport) -> list[Stat]:
        """The goal trained for, with z = B(goal): exact over every start."""
        final, initial = report.snapshots["final"], report.snapshots.get("initial")
        task = report.task_id or "training goal"
        return [
            Stat(
                "Goal success",
                _percent(final.rollout_success_fraction),
                f"{task}, z = B(goal), every start within {report.rollout_horizon} steps",
                _points(final.rollout_success_fraction, initial.rollout_success_fraction)
                if initial
                else None,
            ),
            Stat(
                "Optimal actions",
                _percent(final.optimal_action_fraction),
                f"{task}, cells whose action is optimal",
                _points(final.optimal_action_fraction, initial.optimal_action_fraction)
                if initial
                else None,
            ),
        ]

    @staticmethod
    def _zero_shot(report: ZeroShotSuiteReport) -> list[Stat]:
        protocols = list(report.evidence_protocols)
        main = "dataset_rewards" if "dataset_rewards" in protocols else protocols[0]
        others = [p for p in protocols if p != main]
        overall = report.mean_score["all"]

        def also(values: Mapping[str, float | None]) -> str:
            return " · ".join(f"{p} {values[p]:.2f}" for p in others if values.get(p) is not None)

        stats = []
        if (score := overall.get(main)) is not None:
            detail = _join(f"{len(report.tasks)} tasks, {main}", also(overall))
            stats.append(Stat("Zero-shot score", f"{score:.2f}", detail))
        goals = [
            success
            for t in report.tasks
            if t.family == "goal" and (success := t.outcomes[main].goal_success) is not None
        ]
        if goals:
            stats.append(
                Stat(
                    "Zero-shot goal success",
                    _percent(float(np.mean(goals))),
                    f"mean over {len(goals)} goal tasks, {main}",
                )
            )
        families = [
            (score, family)
            for family, means in report.mean_score.items()
            if family != "all" and (score := means.get(main)) is not None
        ]
        if families:
            score, weakest = min(families)
            detail = _join(f"score {score:.2f}", also(report.mean_score[weakest]))
            stats.append(Stat("Weakest family", weakest, detail))
        return stats
