from dataclasses import dataclass
from math import isfinite
from typing import Any

@dataclass(frozen=True, slots=True)
class Transition:
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
            if isinstance(self.reward, bool) or not isinstance(self.reward, (int, float)):
                raise TypeError("reward must be a scalar number or None")
            
            if not isfinite(self.reward):
                raise ValueError("reward must be finite")
