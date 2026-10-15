"""Diagnostic dataset description (dataset.yaml): the shared, fixed inputs that experiments' views are computed on.

Its construction is documented rather than assumed, because every cross-experiment comparison rests on it.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from rl_lab.contracts.manifest import Model


class Source(Model):
    kind: str                               # e.g. held_out_dataset
    id: str
    note: str | None = None


class Construction(Model):
    method: str                             # e.g. random_windows
    num_samples: int
    window_length: int | None = None
    seed: int | None = None
    constraints: list[str] = Field(default_factory=list)


class Label(Model):
    # reward: a per-task reward value; analyses may derive "rewarded or not" from it.
    kind: Literal["continuous", "categorical", "reward"]
    unit: str | None = None
    classes: int | None = None
    task: str | None = None
    definition: str | None = None


class Grouping(Model):
    column: str                              # samples sharing a value are kept in one fold (e.g. episode_id)


class DatasetSpec(Model):
    id: str
    hash: str
    source: Source
    construction: Construction
    labels: dict[str, Label] = Field(default_factory=dict)
    grouping: Grouping | None = None
    coverage: dict[str, Any] = Field(default_factory=dict)
