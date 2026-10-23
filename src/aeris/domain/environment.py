"""Minimal environment results independent of Gymnasium or torch."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .validation import require_finite_number


@dataclass(frozen=True, slots=True)
class ResetResult:
    observation: Any
    info: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class StepResult:
    """One environment step (the runner assembles a full Transition).

    Known termination flags are required for interactive environments;
    offline datasets may omit them in the separate Transition model.
    """

    observation: Any
    reward: float | None
    terminated: bool
    truncated: bool
    info: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if type(self.terminated) is not bool or type(self.truncated) is not bool:
            raise TypeError("environment end flags must be bool")
        if self.reward is not None:
            require_finite_number("reward", self.reward)
