from typing import Any, Protocol, runtime_checkable

from aeris.domain.environment import ResetResult, StepResult


@runtime_checkable
class Environment(Protocol):
    def reset(self, *, seed: int | None = None) -> ResetResult: ...

    def step(self, action: Any) -> StepResult: ...

    def close(self) -> None: ...
