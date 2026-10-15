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

    # --- Export to rl_lab bundles. Defaults are conservative; algorithms override them. ---

    def checkpoint_step(self, run: ResolvedRun, checkpoint: Path) -> tuple[int, str]:
        """Training step of a checkpoint and its unit: gradient_steps, env_steps or agent_steps."""
        raise NotImplementedError(f"{self.spec.name} does not report checkpoint steps")

    def view_protocol(self, run: ResolvedRun, view: str, info: dict[str, Any]) -> dict[str, Any]:
        """How an extracted view was computed (rl_lab Protocol fields). `info` is extract_info.json plus
        `window_length` of the probe set."""
        return {"input": "unknown", "note": f"{self.spec.name} does not describe its extraction protocol"}

    def budget(self, run: ResolvedRun) -> dict[str, int | None]:
        """Training budget with explicit units (gradient_steps, env_steps, agent_steps)."""
        return {key: value for key, value in self.experience(run).items() if key in ("gradient_steps", "env_steps")}

    def native_metrics(self, run: ResolvedRun) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """(training rows, evaluation rows) translated from the algorithm's own logs.

        Training rows: training_step, step_unit, metric, value. Evaluation rows additionally have task.
        """
        return [], []
