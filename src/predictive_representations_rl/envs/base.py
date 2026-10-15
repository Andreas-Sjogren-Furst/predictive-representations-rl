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
    # Files (globs relative to the project root) that define the tasks' rewards and terminations, as every
    # algorithm runs them; their fingerprint lets comparisons check that runs optimised the same tasks.
    task_definition: tuple[str, ...] = ()


def task_definition_fingerprint(env: EnvSpec, root: Path) -> dict[str, str] | None:
    """sha256 over the task-definition files (relative path + contents), or None if the env declares none."""
    import hashlib

    paths = sorted({path for pattern in env.task_definition for path in root.glob(pattern)})
    if not paths:
        return None
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(root)).encode() + b"\0" + path.read_bytes() + b"\0")
    return {"source": ", ".join(env.task_definition), "fingerprint": "sha256:" + digest.hexdigest()}
