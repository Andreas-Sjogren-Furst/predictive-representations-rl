"""Summaries across the runs in a directory: group by a recorded value, aggregate seeds."""

from __future__ import annotations

from pathlib import Path

from aeris.analysis.seeds import REPORT_TYPES, GroupSummary, key_metrics, summarize_groups
from aeris.data.run_catalog import list_runs
from aeris.data.run_directory import RunDirectory


def summarize_runs(root: str | Path, *, by: str = "name") -> list[GroupSummary]:
    """Mean ± std of every completed run's headline numbers, per group of ``by``.

    Runs that are still running or failed are left out, so a group's n counts
    only finished seeds.
    """
    rows = []
    for summary in list_runs(root):
        if summary.status != "completed":
            continue
        run = RunDirectory(summary.directory)
        reports = [r for t in REPORT_TYPES if (r := run.read_report(t)) is not None]
        metrics = key_metrics(list(run.read_evaluations()), reports)
        rows.append((run.root.name, run.read_manifest(), metrics))
    return summarize_groups(rows, by=by)
