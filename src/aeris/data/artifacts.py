"""NumPy file formats for derived artifacts: policy snapshots and transition datasets.

Format checks (array keys, shapes, dtypes, fingerprints) live here so that
consumers receive typed domain records and never inspect raw arrays.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np

from aeris.domain.artifacts import PolicySnapshot
from aeris.domain.experience import Transition
from aeris.domain.serialization import to_json_data


def file_sha256(path: str | Path) -> str:
    return sha256(Path(path).read_bytes()).hexdigest()


def write_policy_snapshot_npz(path: str | Path, snapshot: PolicySnapshot) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        positions=np.asarray(snapshot.states, dtype=np.int64),
        actions=np.asarray(snapshot.actions, dtype=np.int64),
    )
    return path


def read_policy_snapshot_npz(path: str | Path) -> PolicySnapshot:
    with np.load(path, allow_pickle=False) as data:
        states = data["positions"]
        actions = data["actions"]
    if states.ndim != 2 or states.dtype.kind not in "iu":
        raise ValueError(f"{path}: snapshot positions must be an integer array [states, dims]")
    if actions.shape != (len(states),) or actions.dtype.kind not in "iu":
        raise ValueError(f"{path}: snapshot actions must be an integer vector [states]")
    return PolicySnapshot(states=states.tolist(), actions=actions.tolist())


def write_transitions_npz(path: str | Path, transitions: Sequence[Transition]) -> str:
    """Save vector observations and integer actions; return the file's SHA-256."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        observations=np.stack([t.observation for t in transitions]),
        actions=np.array([t.action for t in transitions], dtype=np.int64),
        next_observations=np.stack([t.next_observation for t in transitions]),
    )
    return file_sha256(path)


class TransitionFile:
    """A fixed ``.npz`` transition dataset, verified against its recorded fingerprint.

    Implements the ``TransitionSource`` contract. Rewards and episode
    boundaries are not stored, so they are ``None`` (unknown) on read.
    """

    def __init__(self, path: str | Path, *, expected_sha256: str | None = None):
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"transition dataset is missing: {self.path}")
        if expected_sha256 and file_sha256(self.path) != expected_sha256:
            raise ValueError(f"{self.path}: dataset fingerprint differs from run manifest")
        with np.load(self.path, allow_pickle=False) as saved:
            self.observations = saved["observations"]
            self.actions = saved["actions"]
            self.next_observations = saved["next_observations"]
        if (
            self.observations.ndim != 2
            or self.next_observations.shape != self.observations.shape
            or self.actions.shape != (len(self.observations),)
            or self.actions.dtype.kind not in "iu"
        ):
            raise ValueError(
                f"{self.path}: expected observations [N, D], next_observations [N, D] "
                "and integer actions [N]"
            )

    def __len__(self) -> int:
        return len(self.actions)

    def iter_transitions(self) -> Iterator[Transition]:
        for obs, action, nxt in zip(
            self.observations, self.actions, self.next_observations, strict=True
        ):
            yield Transition(observation=obs, action=int(action), next_observation=nxt)


METADATA_KEY = "__metadata__"


def write_arrays_npz(
    path: str | Path, arrays: Mapping[str, np.ndarray], metadata: Mapping[str, Any]
) -> Path:
    """Save named arrays plus JSON metadata in one compressed ``.npz`` (no pickling)."""
    if METADATA_KEY in arrays:
        raise ValueError(f"{METADATA_KEY} is reserved for metadata")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # dict[str, Any]: NumPy's stubs would otherwise match the arrays to its allow_pickle keyword.
    payload: dict[str, Any] = {name: np.asarray(value) for name, value in arrays.items()}
    payload[METADATA_KEY] = np.array(json.dumps(to_json_data(metadata), allow_nan=False))
    np.savez_compressed(path, **payload)
    return path


def read_arrays_npz(path: str | Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with np.load(path, allow_pickle=False) as data:
        if METADATA_KEY not in data:
            raise ValueError(f"{path}: not an AERIS array artifact (no metadata)")
        metadata = json.loads(str(data[METADATA_KEY]))
        arrays = {name: data[name] for name in data.files if name != METADATA_KEY}
    return arrays, metadata
