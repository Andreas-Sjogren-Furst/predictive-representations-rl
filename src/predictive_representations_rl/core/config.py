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
    # Optional name for a variant of the same env/algo/mode/task (e.g. "smoke", "lr3e-4"); gets its own run directory.
    variant: str | None = None
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
    # Options passed to the algorithm's extraction script.
    extract: dict[str, Any] = field(default_factory=dict)
    # Cluster resources for `prl run --submit lsf`; see runner.DEFAULT_RESOURCES.
    resources: dict[str, Any] = field(default_factory=dict)


# Keys whose mappings are merged key by key when a config extends another; every other key is replaced.
MERGED_KEYS = ("budget", "overrides", "environment", "extract", "resources")


def read_raw(path: Path, _seen: tuple[Path, ...] = ()) -> dict[str, Any]:
    """A config file's mapping, with `extends: <path relative to this file>` resolved."""
    path = Path(path).resolve()
    if path in _seen:
        raise ValueError(f"{path}: circular extends")

    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a mapping at the top level")

    base_path = raw.pop("extends", None)
    if base_path is None:
        return raw

    base = read_raw(path.parent / base_path, (*_seen, path))
    for key in MERGED_KEYS:
        if key in raw and key in base:
            raw[key] = {**base[key], **(raw[key] or {})}
    return {**base, **raw}


def load_config(path: Path) -> ExperimentConfig:
    raw = read_raw(path)

    known = {f.name for f in fields(ExperimentConfig)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"{path}: unknown keys {sorted(unknown)}; allowed: {sorted(known)}")

    missing = {"env", "algo", "mode"} - set(raw)
    if missing:
        raise ValueError(f"{path}: missing required keys {sorted(missing)}")

    variant = raw.get("variant")
    if variant is not None and (not isinstance(variant, str) or not variant or "/" in variant or variant.startswith(".")):
        raise ValueError(f"{path}: variant must be a plain name, got {variant!r}")

    if "seeds" in raw:
        raw["seeds"] = tuple(raw["seeds"])

    return ExperimentConfig(**raw)
