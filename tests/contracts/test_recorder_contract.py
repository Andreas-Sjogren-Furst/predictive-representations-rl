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
    recorder.write_evaluation(
        EvaluationRecord(
            "run-01",
            "ckpt-1",
            "task-1",
            "zero_shot",
            42,
            2,
            (MetricValue("success", 0.5, "fraction", "mean"),),
        )
    )
    assert recorder.training[0].step_axis == "gradient_updates"
    assert recorder.evaluations[0].checkpoint_id == "ckpt-1"
    assert recorder.manifests["run-01"].experiment.to_dict()["seed"] == 42


class MemoryRun(MemoryRecorder):
    """One in-memory run that is recorded, read back and annotated with artifacts."""

    def __init__(self) -> None:
        super().__init__()
        self.trajectories = []
        self.snapshots = {}
        self.reports = {}

    def write_trajectory(self, record):
        self.trajectories.append(record)

    def read_manifest(self):
        (manifest,) = self.manifests.values()
        return manifest

    def read_training_metrics(self):
        return iter(self.training)

    def read_evaluations(self):
        return iter(self.evaluations)

    def read_trajectories(self):
        return iter(self.trajectories)

    def write_policy_snapshot(self, label, snapshot):
        self.snapshots[label] = snapshot

    def read_policy_snapshot(self, label):
        return self.snapshots.get(label)

    def write_report(self, report):
        self.reports[type(report).ARTIFACT_NAME] = report

    def read_report(self, report_type):
        return self.reports.get(report_type.ARTIFACT_NAME)


def test_reading_and_artifacts_are_separate_capabilities():
    from aeris.contracts import ArtifactStore, RunReader, TrajectoryRecorder
    from aeris.domain import PolicySnapshot, TabularPolicyReport

    run = MemoryRun()
    assert isinstance(run, RunReader)
    assert isinstance(run, ArtifactStore)
    assert isinstance(run, TrajectoryRecorder)
    assert not isinstance(MemoryRecorder(), RunReader)
    snapshot = PolicySnapshot(states=[(1, 1)], actions=[0])
    run.write_policy_snapshot("final", snapshot)
    assert run.read_policy_snapshot("final") == snapshot
    assert run.read_policy_snapshot("initial") is None
    assert run.read_report(TabularPolicyReport) is None
