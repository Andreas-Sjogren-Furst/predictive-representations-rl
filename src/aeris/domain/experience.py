from dataclasses import dataclass
from typing import Any

from .validation import require_finite_number


@dataclass(frozen=True, slots=True)
class Transition:
    """One recorded step. ``None`` marks unknown reward or episode boundaries."""

    observation: Any
    action: Any
    next_observation: Any
    reward: float | None = None
    terminated: bool | None = None
    truncated: bool | None = None

    def __post_init__(self) -> None:
        for name in ("terminated", "truncated"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, bool):
                raise TypeError(f"{name} must be bool or None")
        if self.reward is not None:
            require_finite_number("reward", self.reward)
