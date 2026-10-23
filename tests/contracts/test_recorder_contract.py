from aeris.contracts import RunRecorder
from aeris.domain import (
    ComponentSpec,
    EvaluationRecord,
    ExperimentSpec,
    MetricValue,
    RunManifest,
    TrainingMetricRecord,
    TrainingPlan,
)


class MemoryRecorder:
    def __init__(self) -> None:
        self.manifests: dict[str, RunManifest] = {}
        self.training: list[TrainingMetricRecord] = []
        self.evaluations: list[EvaluationRecord] = []

    def write_manifest(self, manifest: RunManifest) -> None:
        self.manifests[manifest.run_id] = manifest

    def write_training_metrics(self, record: TrainingMetricRecord) -> None:
        self.training.append(record)

    def write_evaluation(self, record: EvaluationRecord) -> None:
        self.evaluations.append(record)


def test_recording_is_independent_of_storage_backend():
    spec = ExperimentSpec(
        name="contract_test",
        agent=ComponentSpec("fake"),
        dataset=ComponentSpec("offline_source"),
        training=TrainingPlan("offline", ComponentSpec("sample_updates")),
        seed=42,
    )
    recorder = MemoryRecorder()
    assert isinstance(recorder, RunRecorder)
    recorder.write_manifest(RunManifest("run-01", spec, provenance={"git_commit": "test"}))
    loss = MetricValue("fb_loss", 0.1, "unitless", "instant")
    recorder.write_training_metrics(TrainingMetricRecord("run-01", "gradient_updates", 10, (loss,)))
    recorder.write_evaluation(EvaluationRecord("run-01", "ckpt-1", "task-1", "zero_shot", 42, 2, (
        MetricValue("success", 0.5, "fraction", "mean"),
    )))
    assert recorder.training[0].step_axis == "gradient_updates"
    assert recorder.evaluations[0].checkpoint_id == "ckpt-1"
    assert recorder.manifests["run-01"].experiment.to_dict()["seed"] == 42
