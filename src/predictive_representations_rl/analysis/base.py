"""Analysis interface and loading of extracted representations.

Analysers see only arrays and per-probe metadata, never algorithm code or native logs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

from rl_lab.analysis.probes import factor_columns  # noqa: F401  (re-exported; lives in rl_lab)


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
