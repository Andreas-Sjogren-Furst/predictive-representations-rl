from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# (task, observation_type) -> the name the algorithm's own code uses for this environment.
NativeName = Callable[[str | None, str], str]


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    observation_type: str
    files: tuple[str, ...]
    how_to_get: str = ""
    # Held-out file that probe sets are cut from (not used for training).
    probe_file: str | None = None

    def missing_files(self) -> list[Path]:
        paths = [Path(file).expanduser() for file in self.files]
        return [path for path in paths if not path.exists()]


@dataclass(frozen=True)
class EnvSpec:
    name: str
    suite: str
    tasks: tuple[str, ...]
    action_type: str
    observation_types: frozenset[str]
    datasets: Mapping[str, DatasetSpec] = field(default_factory=dict)
    native_names: Mapping[str, NativeName] = field(default_factory=dict)
    description: str = ""
    # Ground-truth factors of a batch of flat observations [N, obs_dim], for probing representations.
    factors: Callable[[np.ndarray], dict[str, np.ndarray]] | None = None
    # Dataset whose probe_file the default probe set is cut from.
    probe_dataset: str | None = None
