from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, get_args

from .serialization import JsonRecord, freeze_settings
from .task import TaskEvidenceRef, TaskSpec
from .validation import require_identifier, require_int

TrainingMode = Literal["offline", "online", "hybrid", "none"]


@dataclass(frozen=True, slots=True)
class ComponentSpec:
    name: str
    settings: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require_identifier("component name", self.name)
        object.__setattr__(self, "settings", freeze_settings(self.settings))


@dataclass(frozen=True, slots=True)
class TrainingPlan:
    mode: TrainingMode
    strategy: ComponentSpec | None = None

    def __post_init__(self) -> None:
        if self.mode not in get_args(TrainingMode):
            raise ValueError(f"unknown training mode: {self.mode!r}")
        if (self.mode == "none") != (self.strategy is None):
            raise ValueError("training strategy is required unless mode='none'")


@dataclass(frozen=True, slots=True)
class EvaluationTarget:
    task: TaskSpec
    inference_evidence: TaskEvidenceRef | None = None


@dataclass(frozen=True, slots=True)
class EvaluationPlan:
    protocol: ComponentSpec
    targets: tuple[EvaluationTarget, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "targets", tuple(self.targets))
        if not self.targets:
            raise ValueError("evaluation needs at least one target")
        identities = [(target.task.task_id, target.inference_evidence) for target in self.targets]
        if len(identities) != len(set(identities)):
            raise ValueError("duplicate task/evidence evaluation target")


@dataclass(frozen=True, slots=True)
class ExperimentSpec(JsonRecord):
    name: str
    agent: ComponentSpec
    training: TrainingPlan
    seed: int
    environment: ComponentSpec | None = None
    dataset: ComponentSpec | None = None
    evaluation: EvaluationPlan | None = None
    initial_checkpoint: str | None = None

    def __post_init__(self) -> None:
        require_identifier("experiment name", self.name)
        require_int("seed", self.seed)

        if self.environment is None and self.dataset is None:
            raise ValueError("an environment or a dataset is required")
        if self.training.mode in ("offline", "hybrid") and self.dataset is None:
            raise ValueError("offline/hybrid training requires a dataset")
        if self.training.mode in ("online", "hybrid") and self.environment is None:
            raise ValueError("online/hybrid training requires an environment")
        if self.training.mode == "none" and not self.initial_checkpoint:
            raise ValueError("evaluation-only runs require an initial checkpoint")

        if self.training.mode == "none" and self.evaluation is None:
            raise ValueError("evaluation-only experiments require an evaluation plan")
        if self.initial_checkpoint is not None:
            require_identifier("initial_checkpoint", self.initial_checkpoint)
