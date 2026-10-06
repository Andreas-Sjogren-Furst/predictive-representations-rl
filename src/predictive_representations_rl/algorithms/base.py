from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from predictive_representations_rl.core.runtime import Command, Runtime

# Extraction scripts run inside each algorithm's runtime; they live here but never import this package.
EXTRACT_DIR = Path(__file__).resolve().parents[1] / "extract"

if TYPE_CHECKING:
    from predictive_representations_rl.runner import ResolvedRun

LearningMode = Literal["online", "offline"]

# What a learning mode needs before it can start:
#   self_collected  the algorithm interacts with the environment and fills its own replay buffer
#   dataset         a fixed offline dataset (e.g. ExORL hdf5) given by the environment registry
#   replay          the replay directory of a previous run (e.g. Dreamer passive training)
DataRequirement = Literal["self_collected", "dataset", "replay"]


@dataclass(frozen=True)
class AlgorithmSpec:
    """What an algorithm is. Describes the algorithm as implemented; it is not forced into a mode."""

    name: str
    description: str
    modes: Mapping[LearningMode, DataRequirement]
    action_spaces: frozenset[str]
    observation_types: frozenset[str]
    representations: tuple[str, ...]
    # True if the representation depends on history (recurrent state), so probes need sequences.
    stateful_representation: bool
    # True if training optimises a single task's reward; False if it pretrains task-agnostically.
    task_specific: bool
    runtime: Runtime


class AlgorithmAdapter(ABC):
    """How to run an algorithm: builds its native command line and finds what it wrote. No learning happens here."""

    spec: AlgorithmSpec

    @abstractmethod
    def train_command(self, run: ResolvedRun) -> Command:
        """The algorithm's own training entry point, writing its outputs under `run.native_dir`."""

    @abstractmethod
    def find_checkpoint(self, run: ResolvedRun) -> Path | None:
        """The latest checkpoint the algorithm wrote, if any."""

    @abstractmethod
    def extract_command(self, run: ResolvedRun, checkpoint: Path, probes: Path, out_dir: Path) -> Command:
        """Command that writes <representation>.npz (key `values`) and extract_info.json for a probe set into `out_dir`."""

    @abstractmethod
    def experience(self, run: ResolvedRun) -> dict[str, Any]:
        """How much data and compute the run is configured to use (env steps, dataset, gradient steps)."""
