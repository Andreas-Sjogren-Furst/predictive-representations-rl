from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .serialization import JsonRecord, freeze_settings
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


def task_family(task: TaskSpec) -> str:
    """The task's type from its settings; tasks without one are goal tasks."""
    return str(task.settings.get("type", "goal"))


@dataclass(frozen=True, slots=True)
class TaskSuite(JsonRecord):
    """A named, versioned set of tasks that every agent is evaluated on in the same way."""

    name: str
    description: str
    tasks: tuple[TaskSpec, ...]

    def __post_init__(self) -> None:
        require_identifier("name", self.name)
        object.__setattr__(self, "tasks", tuple(self.tasks))
        if not self.tasks:
            raise ValueError("a task suite needs at least one task")
        ids = [task.task_id for task in self.tasks]
        if len(set(ids)) != len(ids):
            raise ValueError("task ids in a suite must be unique")

    @property
    def families(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(task_family(task) for task in self.tasks))
