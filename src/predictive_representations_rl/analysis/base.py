"""Analysis interface and loading of extracted representations.

Analysers see only arrays and per-probe metadata, never algorithm code or native logs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

# Bookkeeping columns of a probe set that are not properties of the probe state.
NON_FACTORS = frozenset({"source_index", "episode"})
# Integer metadata with at most this many distinct values is treated as a class label.
MAX_CLASSES = 20
# Per-task rewards are probed as "rewarded or not". Tolerance-style rewards decay smoothly and only underflow to
# exactly 0 far from the goal, so a tiny threshold separates near-goal states from the rest.
REWARD_THRESHOLD = 1e-6


class Analyzer(ABC):
    name: str

    @abstractmethod
    def run(self, representations: dict[str, np.ndarray], metadata: dict[str, np.ndarray], output_dir: Path) -> dict[str, Any]:
        """Analyse [N, D] representations against per-probe metadata; write files to output_dir; return a summary."""


def load_extraction(probe_dir: Path) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Representations with one row per probe ([N, D]) and the probe metadata, from probes/<probe set>/."""
    with np.load(probe_dir / "metadata.npz") as data:
        metadata = {key: data[key] for key in data.files}
    num = len(next(iter(metadata.values())))

    representations = {}
    for path in sorted(probe_dir.glob("*.npz")):
        if path.name == "metadata.npz":
            continue
        with np.load(path) as data:
            values = data["values"]
        if values.ndim == 2 and len(values) == num:  # per-probe; skips e.g. FB's per-task latents
            representations[path.stem] = values.astype(np.float64)
    return representations, metadata


def factor_columns(metadata: dict[str, np.ndarray]) -> dict[str, tuple[np.ndarray, str]]:
    """Metadata columns to probe for, each tagged 'categorical' or 'continuous'.

    `reward_<task>` columns become binary `rewarded_<task>` labels (reward > REWARD_THRESHOLD): rewards are mostly 0,
    which makes regression R^2 meaningless. Ordered: environment factors, then rewards, then the step in the episode.
    """
    def order(key: str) -> int:
        return 2 if key == "step" else 1 if key.startswith("reward_") else 0

    factors = {}
    for key, values in sorted(metadata.items(), key=lambda item: order(item[0])):
        if key in NON_FACTORS or values.ndim != 1:
            continue
        if key.startswith("reward_"):
            factors["rewarded_" + key.removeprefix("reward_")] = ((values > REWARD_THRESHOLD).astype(np.int64), "categorical")
            continue
        categorical = np.issubdtype(values.dtype, np.integer) and len(np.unique(values)) <= MAX_CLASSES
        factors[key] = (values, "categorical" if categorical else "continuous")
    return factors
