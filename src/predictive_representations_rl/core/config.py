from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ExperimentConfig:
    env: str
    algo: str
    mode: str
    obs_type: str = "state"
    task: str | None = None
    dataset: str | None = None
    replay_from: str | None = None
    seeds: tuple[int, ...] = (0,)
    budget: dict[str, Any] = field(default_factory=dict)
    # Passed through unchanged to the algorithm's own command line.
    overrides: dict[str, Any] = field(default_factory=dict)
    # Extra environment variables for the algorithm process (e.g. JAX_PLATFORMS: cpu).
    environment: dict[str, str] = field(default_factory=dict)
    # Cluster resources for `prl run --submit lsf`; see runner.DEFAULT_RESOURCES.
    resources: dict[str, Any] = field(default_factory=dict)


def load_config(path: Path) -> ExperimentConfig:
    raw = yaml.safe_load(Path(path).read_text()) or {}

    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a mapping at the top level")

    known = {f.name for f in fields(ExperimentConfig)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"{path}: unknown keys {sorted(unknown)}; allowed: {sorted(known)}")

    missing = {"env", "algo", "mode"} - set(raw)
    if missing:
        raise ValueError(f"{path}: missing required keys {sorted(missing)}")

    if "seeds" in raw:
        raw["seeds"] = tuple(raw["seeds"])

    return ExperimentConfig(**raw)
