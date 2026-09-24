from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .data import Observation, RepresentationDataset, TaskDataset

@dataclass(frozen=True)
class TrainingResult:
    gradient_updates: int
    metrics: dict[str, float] = field(default_factory=dict)

class Policy(ABC):

    @abstractmethod
    def initial_state(self) -> Any:
        raise NotImplementedError

    @abstractmethod
    def act(self, observation: Observation, state: Any, *, deterministic: bool = True) -> tuple[np.ndarray, Any]:
        raise NotImplementedError

@dataclass(frozen=True)
class AdaptationResult:
    policy: Policy
    gradient_updates: int
    metrics: dict[str, float] = field(default_factory=dict)

class PredictiveRLBackend(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def fit_representation(self, dataset: RepresentationDataset) -> TrainingResult:
        raise NotImplementedError

    @abstractmethod
    def adapt_task(self, dataset: TaskDataset) -> AdaptationResult:
        raise NotImplementedError

    @abstractmethod
    def save_representation(self, path: Path) -> None:
        raise NotImplementedError

    @abstractmethod
    def load_representation(self, path: Path) -> None:
        raise NotImplementedError
    