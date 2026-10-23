from typing import Protocol, runtime_checkable

from aeris.domain.results import EvaluationRecord, RunManifest, TrainingMetricRecord


@runtime_checkable
class RunRecorder(Protocol):
    def write_manifest(self, manifest: RunManifest) -> None: ...

    def write_training_metrics(self, record: TrainingMetricRecord) -> None: ...

    def write_evaluation(self, record: EvaluationRecord) -> None: ...
