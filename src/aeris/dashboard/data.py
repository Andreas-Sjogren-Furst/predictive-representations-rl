"""Everything a panel may read about a run, loaded once through RunDirectory."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from aeris.agents.fb.reports import FBCheckpointReport
from aeris.agents.fb.representations import FBRepresentations
from aeris.data.run_directory import RunDirectory
from aeris.domain.artifacts import PolicySnapshot
from aeris.domain.reports import AnalysisReport, TabularPolicyReport, ZeroShotReport
from aeris.domain.results import EvaluationRecord, RunManifest, TrainingMetricRecord
from aeris.domain.trajectory import EvaluationTrajectory

REPORT_TYPES: tuple[type[AnalysisReport], ...] = (
    TabularPolicyReport,
    FBCheckpointReport,
    ZeroShotReport,
)
SNAPSHOT_LABELS = ("initial", "final")


@dataclass(frozen=True)
class RunData:
    """An immutable view of one run's artifacts; panels receive these, never files."""

    label: str  # directory name: unique, unlike run_id, which is derived from the seed
    root: Path
    manifest: RunManifest
    training: tuple[TrainingMetricRecord, ...]
    evaluations: tuple[EvaluationRecord, ...]
    trajectories: tuple[EvaluationTrajectory, ...]
    snapshots: Mapping[str, PolicySnapshot]
    reports: Mapping[str, AnalysisReport]  # keyed by the report class name
    representations: FBRepresentations | None = None

    def report(self, report_type: type[AnalysisReport]) -> AnalysisReport | None:
        return self.reports.get(report_type.__name__)

    def setting(self, path: str) -> object:
        """A component setting by dotted path, e.g. ``agent.hidden_dim`` or ``dataset.origin``."""
        component, key = path.split(".", 1)
        spec = getattr(self.manifest.experiment, component, None)
        return None if spec is None else spec.settings.get(key)


def load_run(root: str | Path) -> RunData:
    store = RunDirectory(root)
    reports = {
        report_type.__name__: report
        for report_type in REPORT_TYPES
        if (report := store.read_report(report_type)) is not None
    }
    return RunData(
        label=store.root.name,
        root=store.root,
        manifest=store.read_manifest(),
        training=tuple(store.read_training_metrics()),
        evaluations=tuple(store.read_evaluations()),
        trajectories=tuple(store.read_trajectories()),
        snapshots={
            label: snapshot
            for label in SNAPSHOT_LABELS
            if (snapshot := store.read_policy_snapshot(label)) is not None
        },
        reports=reports,
        representations=(
            FBRepresentations.from_arrays(*saved)
            if (saved := store.read_arrays(FBRepresentations.ARTIFACT_NAME)) is not None
            else None
        ),
    )


def run_stamp(root: str | Path) -> tuple[tuple[str, int], ...]:
    """File names and modification times: a cache key that changes when the run changes."""
    root = Path(root)
    files = sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix in {".json", ".jsonl", ".npz"}
    )
    return tuple((str(p.relative_to(root)), p.stat().st_mtime_ns) for p in files)
