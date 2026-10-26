from collections.abc import Iterator, Mapping
from typing import Any, Protocol, TypeVar, runtime_checkable

from aeris.domain.artifacts import PolicySnapshot
from aeris.domain.reports import AnalysisReport
from aeris.domain.results import EvaluationRecord, RunManifest, TrainingMetricRecord
from aeris.domain.trajectory import EvaluationTrajectory

ReportT = TypeVar("ReportT", bound=AnalysisReport)


@runtime_checkable
class RunRecorder(Protocol):
    def write_manifest(self, manifest: RunManifest) -> None: ...

    def write_training_metrics(self, record: TrainingMetricRecord) -> None: ...

    def write_evaluation(self, record: EvaluationRecord) -> None: ...


@runtime_checkable
class TrajectoryRecorder(Protocol):
    def write_trajectory(self, record: EvaluationTrajectory) -> None: ...


@runtime_checkable
class RunReader(Protocol):
    """Typed read access to everything a recorder wrote for one run."""

    def read_manifest(self) -> RunManifest: ...

    def read_training_metrics(self) -> Iterator[TrainingMetricRecord]: ...

    def read_evaluations(self) -> Iterator[EvaluationRecord]: ...

    def read_trajectories(self) -> Iterator[EvaluationTrajectory]: ...


@runtime_checkable
class ArtifactStore(Protocol):
    """Derived per-run artifacts: policy snapshots and analysis reports.

    ``label`` names a snapshot within a run (for example ``"initial"`` or
    ``"final"``). Reads return ``None`` when the artifact was never written.
    """

    def write_policy_snapshot(self, label: str, snapshot: PolicySnapshot) -> None: ...

    def read_policy_snapshot(self, label: str) -> PolicySnapshot | None: ...

    def write_report(self, report: AnalysisReport) -> None: ...

    def read_report(self, report_type: type[ReportT]) -> ReportT | None: ...


@runtime_checkable
class ArrayStore(Protocol):
    """Named numeric arrays with JSON metadata, for artifacts too large for JSON reports.

    Arrays are left untyped here so contracts stay free of NumPy; algorithm
    packages wrap them in typed records (e.g. ``agents.fb.FBRepresentations``).
    """

    def write_arrays(
        self, name: str, arrays: Mapping[str, Any], metadata: Mapping[str, Any]
    ) -> Any: ...

    def read_arrays(self, name: str) -> tuple[dict[str, Any], dict[str, Any]] | None: ...
