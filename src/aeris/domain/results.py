from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, get_args

from .experiment import ExperimentSpec
from .serialization import JsonRecord, freeze_settings
from .task import TaskEvidenceRef
from .validation import require_finite_number, require_identifier, require_int

RunStatus = Literal["created", "running", "completed", "failed", "interrupted"]


@dataclass(frozen=True, slots=True)
class RunManifest(JsonRecord):
    run_id: str
    experiment: ExperimentSpec
    status: RunStatus = "created"
    provenance: Mapping[str, Any] = field(default_factory=dict)
    selected_checkpoint: str | None = None
    error: str | None = None

    def setting(self, path: str) -> Any:
        """A recorded value by path: ``name``, ``seed``, or ``<component>.<key>``.

        Components: ``agent``, ``environment`` and ``dataset`` (their settings),
        ``training`` (the strategy's settings) and ``evaluation`` (the
        protocol's settings). Missing values are None.
        """
        experiment = self.experiment
        if path in ("name", "seed"):
            return getattr(experiment, path)
        component, _, key = path.partition(".")
        spec = {
            "agent": experiment.agent,
            "environment": experiment.environment,
            "dataset": experiment.dataset,
            "training": experiment.training.strategy,
            "evaluation": experiment.evaluation.protocol if experiment.evaluation else None,
        }.get(component)
        return None if spec is None or not key else spec.settings.get(key)

    def __post_init__(self) -> None:
        require_identifier("run_id", self.run_id)
        if self.status not in get_args(RunStatus):
            raise ValueError(f"unknown run status: {self.status!r}")
        object.__setattr__(self, "provenance", freeze_settings(self.provenance))


@dataclass(frozen=True, slots=True)
class MetricValue:
    name: str
    value: float
    unit: str
    aggregation: str

    def __post_init__(self) -> None:
        require_identifier("metric name", self.name)
        require_identifier("metric unit", self.unit)
        require_identifier("metric aggregation", self.aggregation)
        require_finite_number("metric value", self.value)


@dataclass(frozen=True, slots=True)
class EvaluationRecord(JsonRecord):
    run_id: str
    checkpoint_id: str
    task_id: str
    protocol_id: str
    seed: int
    episode_count: int
    metrics: tuple[MetricValue, ...]
    inference_evidence: TaskEvidenceRef | None = None

    def __post_init__(self) -> None:
        for name in ("run_id", "checkpoint_id", "task_id", "protocol_id"):
            require_identifier(name, getattr(self, name))

        require_int("seed", self.seed)
        require_int("episode_count", self.episode_count, minimum=1)
        object.__setattr__(self, "metrics", tuple(self.metrics))
        if len({m.name for m in self.metrics}) != len(self.metrics):
            raise ValueError("duplicate metric names in one evaluation")


@dataclass(frozen=True, slots=True)
class TrainingMetricRecord(JsonRecord):
    run_id: str
    step_axis: str
    step: int
    metrics: tuple[MetricValue, ...]

    def __post_init__(self) -> None:
        require_identifier("run_id", self.run_id)
        require_identifier("step_axis", self.step_axis)

        require_int("step", self.step)
        object.__setattr__(self, "metrics", tuple(self.metrics))

        if len({m.name for m in self.metrics}) != len(self.metrics):
            raise ValueError("duplicate metric names in one training record")
