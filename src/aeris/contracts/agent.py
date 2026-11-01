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


@runtime_checkable
class ZeroShotAgent(Agent, TaskConditioner[EvidenceT], Protocol[EvidenceT]):
    """An agent that acts on a condition it infers from task evidence, without learning."""


@runtime_checkable
class FeatureExtractor(Protocol):
    """Internal activations of an agent's networks, for analysis.

    ``feature_layers`` names every layer in network order (input side first) and
    says whether it depends on the task condition. ``extract_features`` returns
    one array [N, k] per layer, a row per observation; conditioned layers use
    ``condition``, and unconditioned ones ignore it. A layer named ``"Q"``, if
    present, holds the action values [N, A] the agent acts greedily on.
    """

    def feature_layers(self) -> Mapping[str, bool]: ...

    def extract_features(
        self, observations: Any, condition: Any | None = None
    ) -> Mapping[str, Any]: ...
