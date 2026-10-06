"""Fixed probe inputs that every algorithm's representation is evaluated on.

A probe is a short trajectory window ending at the probe state. Stateless representations (FB) use only the
final step; history-dependent ones (Dreamer's RSSM) are filtered over the whole window.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

Factors = Callable[[np.ndarray], dict[str, np.ndarray]]


@dataclass(frozen=True)
class ProbeSet:
    observations: np.ndarray        # [N, T, obs_dim]
    actions: np.ndarray             # [N, T, act_dim]; actions[:, t] is taken at observations[:, t]
    metadata: dict[str, np.ndarray]  # per probe, describing the final state

    def __post_init__(self) -> None:
        if self.observations.ndim != 3 or self.actions.ndim != 3:
            raise ValueError("observations and actions must be [N, T, dim]")
        if self.observations.shape[:2] != self.actions.shape[:2]:
            raise ValueError("observations and actions must share [N, T]")
        for key, value in self.metadata.items():
            if len(value) != len(self.observations):
                raise ValueError(f"metadata {key!r} has {len(value)} rows, expected {len(self.observations)}")

    @property
    def final_observations(self) -> np.ndarray:
        return self.observations[:, -1]

    @property
    def final_actions(self) -> np.ndarray:
        return self.actions[:, -1]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {f"meta/{key}": value for key, value in self.metadata.items()}
        np.savez(path, observations=self.observations, actions=self.actions, **meta)

    @classmethod
    def load(cls, path: Path) -> ProbeSet:
        with np.load(path) as data:
            metadata = {key.removeprefix("meta/"): data[key] for key in data.files if key.startswith("meta/")}
            return cls(observations=data["observations"], actions=data["actions"], metadata=metadata)


def sample_windows(episode_starts: np.ndarray, size: int, num: int, length: int, rng: np.random.Generator) -> np.ndarray:
    """Start indices of `num` windows of `length` steps that stay inside one episode.

    The last step of each episode is excluded because its stored action is a placeholder.
    """
    starts = np.flatnonzero(episode_starts)
    ends = np.append(starts[1:], size) - 1  # index of each episode's last step

    valid = np.concatenate([np.arange(start, end - length + 1) for start, end in zip(starts, ends) if end - start >= length])
    if len(valid) < num:
        raise ValueError(f"only {len(valid)} windows of length {length} available, asked for {num}")

    return np.sort(rng.choice(valid, size=num, replace=False))


def build_from_exorl(path: Path, num: int, length: int, seed: int, factors: Factors | None) -> ProbeSet:
    """Cut probe windows from an ExORL hdf5 file (observations, actions, resets, rewards/<task>)."""
    import h5py

    with h5py.File(Path(path).expanduser(), "r") as file:
        observations = file["observations"][:]
        actions = file["actions"][:]
        resets = file["resets"][:]
        rewards = {task: file["rewards"][task][:] for task in file["rewards"]} if "rewards" in file else {}

    starts = sample_windows(resets, len(observations), num, length, np.random.default_rng(seed))
    window = starts[:, None] + np.arange(length)[None]
    final = window[:, -1]

    episode = np.cumsum(resets) - 1
    episode_start = np.flatnonzero(resets)[episode]
    metadata = {
        "source_index": final,
        "episode": episode[final],
        "step": final - episode_start[final],
        **{f"reward_{task}": values[final] for task, values in rewards.items()},
        **(factors(observations[final]) if factors else {}),
    }

    return ProbeSet(observations=observations[window], actions=actions[window], metadata=metadata)
