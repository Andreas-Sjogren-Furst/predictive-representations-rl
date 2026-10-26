from typing import Any, Protocol, runtime_checkable

from aeris.domain.artifacts import DiscreteState
from aeris.domain.environment import ResetResult, StepResult
from aeris.domain.spaces import SpaceSpec
from aeris.domain.task import TaskSpec


@runtime_checkable
class Environment(Protocol):
    def reset(self, *, seed: int | None = None) -> ResetResult: ...

    def step(self, action: Any) -> StepResult: ...

    def close(self) -> None: ...


@runtime_checkable
class TransitionTable(Protocol):
    """Exact deterministic dynamics over enumerated discrete states.

    ``next_states`` is an integer array ``[states, actions]`` of successor
    indices into ``states``.
    """

    @property
    def states(self) -> tuple[DiscreteState, ...]: ...

    @property
    def next_states(self) -> Any: ...


@runtime_checkable
class TabularModel(Protocol):
    """An environment that can expose its exact model for oracle analyses.

    Only small diagnostic environments implement this; learning code must not
    depend on it.
    """

    @property
    def spec(self) -> SpaceSpec: ...

    def tabular_dynamics(self) -> TransitionTable: ...

    def observation_for(self, state: DiscreteState) -> Any: ...

    def goal_from_task(self, task: TaskSpec) -> DiscreteState: ...

    def state_rewards(self, task: TaskSpec) -> Any:
        """Reward r(s) for being in each state, aligned with ``tabular_dynamics().states``."""
        ...
