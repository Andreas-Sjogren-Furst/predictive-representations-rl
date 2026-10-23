from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .serialization import freeze_settings
from .validation import require_identifier


@dataclass(frozen=True, slots=True)
class TaskSpec:
    task_id: str
    settings: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require_identifier("task_id", self.task_id)
        object.__setattr__(self, "settings", freeze_settings(self.settings))


@dataclass(frozen=True, slots=True)
class TaskEvidenceRef:
    source_id: str
    split: str

    def __post_init__(self) -> None:
        require_identifier("source_id", self.source_id)
        require_identifier("split", self.split)
