from collections.abc import Mapping
from typing import Any, Protocol, TypeVar, runtime_checkable

from aeris.domain.agent import ActionRequest, ActionResult
from aeris.domain.results import MetricValue
from aeris.domain.task import TaskSpec

BatchT = TypeVar("BatchT", contravariant=True)
EvidenceT = TypeVar("EvidenceT", contravariant=True)


@runtime_checkable
class Agent(Protocol):
    def act(self, request: ActionRequest) -> ActionResult: ...


@runtime_checkable
class TrainableAgent(Agent, Protocol[BatchT]):
    def update(self, batch: BatchT) -> tuple[MetricValue, ...]: ...


@runtime_checkable
class Checkpointable(Protocol):
    def state_dict(self) -> Mapping[str, Any]: ...

    def load_state_dict(self, state: Mapping[str, Any]) -> None: ...


@runtime_checkable
class TaskConditioner(Protocol[EvidenceT]):
    def infer_condition(self, task: TaskSpec, evidence: EvidenceT) -> Any: ...
