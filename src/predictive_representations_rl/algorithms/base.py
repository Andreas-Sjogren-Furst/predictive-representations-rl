from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from predictive_representations_rl.core.runtime import Runtime

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
