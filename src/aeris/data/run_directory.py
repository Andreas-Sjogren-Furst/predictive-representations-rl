"""A run stored as a local directory of JSON, JSON Lines and NumPy files.

Layout::

    manifest.json            RunManifest (rewritten atomically)
    training_metrics.jsonl   TrainingMetricRecord per line
    evaluations.jsonl        EvaluationRecord per line
    trajectories.jsonl       EvaluationTrajectory per line (optional)
    policy_<label>.npz       PolicySnapshot
    dataset.npz              fixed training transitions (see data.artifacts)
    checkpoint.pt            trusted agent checkpoint (written by the agent package)
    analysis/<report>.json   AnalysisReport subclasses
    analysis/<name>.npz      named array artifacts with JSON metadata (e.g. representations)
    figures/                 rendered plots

Implements the RunRecorder, TrajectoryRecorder, RunReader, ArtifactStore and ArrayStore
contracts.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any, TypeVar

from aeris.data.artifacts import (
    read_arrays_npz,
    read_policy_snapshot_npz,
    write_arrays_npz,
    write_policy_snapshot_npz,
)
from aeris.domain.artifacts import PolicySnapshot
from aeris.domain.reports import AnalysisReport
from aeris.domain.results import EvaluationRecord, RunManifest, TrainingMetricRecord
from aeris.domain.serialization import JsonRecord, to_json_data
from aeris.domain.trajectory import EvaluationTrajectory

RecordT = TypeVar("RecordT", bound=JsonRecord)
ReportT = TypeVar("ReportT", bound=AnalysisReport)

MANIFEST = "manifest.json"
TRAINING_METRICS = "training_metrics.jsonl"
EVALUATIONS = "evaluations.jsonl"
TRAJECTORIES = "trajectories.jsonl"


class RunDirectory:
    """One run per directory: atomic manifest writes, append-only record streams."""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    # Well-known artifact locations for components that own their file format.
    @property
    def manifest_path(self) -> Path:
        return self.root / MANIFEST

    @property
    def dataset_path(self) -> Path:
        return self.root / "dataset.npz"

    @property
    def checkpoint_path(self) -> Path:
        return self.root / "checkpoint.pt"

    @property
    def figures_dir(self) -> Path:
        return self.root / "figures"

    def exists(self) -> bool:
        return self.manifest_path.is_file()

    # RunRecorder / TrajectoryRecorder
    def write_manifest(self, manifest: RunManifest) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        temporary = self.root / f"{MANIFEST}.tmp"
        with temporary.open("w", encoding="utf8") as stream:
            json.dump(manifest.to_dict(), stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(self.manifest_path)

    def write_training_metrics(self, record: TrainingMetricRecord) -> None:
        self._append(TRAINING_METRICS, record.to_dict())

    def write_evaluation(self, record: EvaluationRecord) -> None:
        self._append(EVALUATIONS, record.to_dict())

    def write_trajectory(self, record: EvaluationTrajectory) -> None:
        """Append a compact diagnostic episode (not meant for image observations)."""
        self._append(TRAJECTORIES, to_json_data(record))

    # RunReader
    def read_manifest(self) -> RunManifest:
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"no {MANIFEST} found in {self.root}")
        return _read_record(self.manifest_path, RunManifest)

    def read_training_metrics(self) -> Iterator[TrainingMetricRecord]:
        return self._read_records(TRAINING_METRICS, TrainingMetricRecord)

    def read_evaluations(self) -> Iterator[EvaluationRecord]:
        return self._read_records(EVALUATIONS, EvaluationRecord)

    def read_trajectories(self) -> Iterator[EvaluationTrajectory]:
        return self._read_records(TRAJECTORIES, EvaluationTrajectory)

    # ArtifactStore
    def policy_snapshot_path(self, label: str) -> Path:
        return self.root / f"policy_{label}.npz"

    def write_policy_snapshot(self, label: str, snapshot: PolicySnapshot) -> None:
        write_policy_snapshot_npz(self.policy_snapshot_path(label), snapshot)

    def read_policy_snapshot(self, label: str) -> PolicySnapshot | None:
        path = self.policy_snapshot_path(label)
        return read_policy_snapshot_npz(path) if path.exists() else None

    def report_path(self, report_type: type[AnalysisReport]) -> Path:
        return self.root / "analysis" / report_type.ARTIFACT_NAME

    def write_report(self, report: AnalysisReport) -> Path:
        path = self.report_path(type(report))
        path.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(report.to_dict(), indent=2, allow_nan=False) + "\n"
        path.write_text(text, encoding="utf-8")
        return path

    def read_report(self, report_type: type[ReportT]) -> ReportT | None:
        path = self.report_path(report_type)
        return _read_record(path, report_type) if path.exists() else None

    # ArrayStore
    def arrays_path(self, name: str) -> Path:
        return self.root / "analysis" / f"{name}.npz"

    def write_arrays(
        self, name: str, arrays: Mapping[str, Any], metadata: Mapping[str, Any]
    ) -> Path:
        return write_arrays_npz(self.arrays_path(name), arrays, metadata)

    def read_arrays(self, name: str) -> tuple[dict[str, Any], dict[str, Any]] | None:
        path = self.arrays_path(name)
        return read_arrays_npz(path) if path.exists() else None

    # Helpers
    def _append(self, filename: str, obj: dict[str, Any]) -> None:
        # Serialize before opening so a rejected record never leaves a partial line.
        line = json.dumps(obj, sort_keys=True, allow_nan=False) + "\n"
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / filename).open("a", encoding="utf8") as stream:
            stream.write(line)

    def _read_records(self, filename: str, record_type: type[RecordT]) -> Iterator[RecordT]:
        """Stream typed records; a missing file is an empty stream."""
        path = self.root / filename
        if not path.exists():
            return
        with path.open(encoding="utf-8") as stream:
            for lineno, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON in {path}:{lineno}: {exc}") from exc
                try:
                    yield record_type.from_dict(data)
                except ValueError as exc:
                    raise ValueError(f"{path}:{lineno}: {exc}") from exc


def _read_record(path: Path, record_type: type[RecordT]) -> RecordT:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path}: {exc}") from exc
    try:
        return record_type.from_dict(data)
    except ValueError as exc:
        raise ValueError(f"{path}: {exc}") from exc
