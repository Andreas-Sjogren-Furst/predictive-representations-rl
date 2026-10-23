from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ActionRequest:
    observation: Any
    condition: Any | None = None
    policy_state: Any | None = None
    deterministic: bool = True


@dataclass(frozen=True, slots=True)
class ActionResult:
    action: Any
    next_policy_state: Any | None = None
