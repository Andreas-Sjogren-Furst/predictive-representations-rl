"""The activations of every layer of an agent's networks, saved for analysis.

Any agent that implements the ``FeatureExtractor`` contract can fill this
record: layers are named in network order, and conditioned layers (those that
depend on the task vector z) are saved for a list of conditions. The record
only needs NumPy, so dashboards read it without torch.

Shapes: S states, C conditions, k units of a layer, d dimensions of z.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from aeris.domain.artifacts import DiscreteState

# Kinds of condition, in the order the export lists them.
REFERENCE = "reference"  # the run's training goal, z = B(goal)
LENGTH = "length"  # the reference direction rescaled to another length
TASK = "task"  # z inferred from reward samples for one suite task
INTERPOLATION = "interpolation"  # between the reference and another goal's z
CONDITION_KINDS = (REFERENCE, LENGTH, TASK, INTERPOLATION)


@dataclass(frozen=True)
class FeatureCondition:
    label: str
    kind: str
    tasks: tuple[str, ...]  # suite task ids this condition's policy is scored on
    length: float  # |z|
    position: float | None = None  # LENGTH: |z|; INTERPOLATION: t from the reference (0) on

    def __post_init__(self) -> None:
        if self.kind not in CONDITION_KINDS:
            raise ValueError(f"unknown condition kind {self.kind!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "kind": self.kind,
            "tasks": list(self.tasks),
            "length": self.length,
            "position": self.position,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> FeatureCondition:
        return cls(
            label=data["label"],
            kind=data["kind"],
            tasks=tuple(data["tasks"]),
            length=float(data["length"]),
            position=None if data["position"] is None else float(data["position"]),
        )


@dataclass(frozen=True, eq=False)
class NetworkFeatures:
    ARTIFACT_NAME: ClassVar[str] = "network_features_v1"

    run_id: str
    checkpoint: str
    gamma: float
    states: tuple[DiscreteState, ...]
    layers: tuple[str, ...]  # network order, input side first
    conditioned: frozenset[str]  # layers that depend on z
    conditions: tuple[FeatureCondition, ...]
    condition_vectors: np.ndarray  # [C, d]
    features: Mapping[str, np.ndarray]  # unconditioned [S, k]; conditioned [C, S, k]

    def __post_init__(self) -> None:
        s, c = len(self.states), len(self.conditions)
        if set(self.features) != set(self.layers):
            raise ValueError("features must hold exactly the listed layers")
        if not self.conditioned <= set(self.layers):
            raise ValueError("conditioned layers must be listed layers")
        if self.condition_vectors.ndim != 2 or len(self.condition_vectors) != c:
            raise ValueError("condition_vectors must have one row per condition")
        for name, values in self.features.items():
            shape = (c, s) if name in self.conditioned else (s,)
            if values.ndim != len(shape) + 1 or values.shape[: len(shape)] != shape:
                raise ValueError(f"layer {name!r} has shape {values.shape}, not {shape} + (k,)")
        if len({condition.label for condition in self.conditions}) != c:
            raise ValueError("condition labels must be unique")

    @property
    def reference(self) -> int:
        """Index of the reference condition (the training goal's z), or 0."""
        return next((i for i, c in enumerate(self.conditions) if c.kind == REFERENCE), 0)

    def condition_index(self, label: str) -> int:
        return [c.label for c in self.conditions].index(label)

    def layer(self, name: str, condition: int | None = None) -> np.ndarray:
        """The layer's activations [S, k]; conditioned layers need a condition index."""
        values = self.features[name]
        if name not in self.conditioned:
            return values
        return values[self.reference if condition is None else condition]

    def actions(self, condition: int) -> np.ndarray | None:
        """The greedy action in every state, from the ``Q`` layer; None without one."""
        if "Q" not in self.features:
            return None
        return np.asarray(self.layer("Q", condition).argmax(axis=1))

    def to_arrays(self) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        arrays = {
            f"layer:{name}": np.asarray(values, dtype=np.float32)
            for name, values in self.features.items()
        }
        arrays["condition_vectors"] = np.asarray(self.condition_vectors, dtype=np.float32)
        metadata = {
            "schema_version": 1,
            "run_id": self.run_id,
            "checkpoint": self.checkpoint,
            "gamma": self.gamma,
            "states": [list(s) for s in self.states],
            "layers": list(self.layers),
            "conditioned": [name for name in self.layers if name in self.conditioned],
            "conditions": [condition.to_dict() for condition in self.conditions],
        }
        return arrays, metadata

    @classmethod
    def from_arrays(
        cls, arrays: Mapping[str, np.ndarray], metadata: Mapping[str, Any]
    ) -> NetworkFeatures:
        if metadata.get("schema_version") != 1:
            raise ValueError(
                f"unsupported {cls.ARTIFACT_NAME} schema: {metadata.get('schema_version')!r}"
            )
        layers = tuple(metadata["layers"])
        return cls(
            run_id=metadata["run_id"],
            checkpoint=metadata["checkpoint"],
            gamma=float(metadata["gamma"]),
            states=tuple(tuple(int(x) for x in s) for s in metadata["states"]),
            layers=layers,
            conditioned=frozenset(metadata["conditioned"]),
            conditions=tuple(FeatureCondition.from_dict(c) for c in metadata["conditions"]),
            condition_vectors=np.asarray(arrays["condition_vectors"]),
            features={name: np.asarray(arrays[f"layer:{name}"]) for name in layers},
        )
