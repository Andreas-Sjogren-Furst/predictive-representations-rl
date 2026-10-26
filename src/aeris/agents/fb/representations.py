"""FB representations of every state, saved as an array artifact for inspection.

Shapes: S states, G goals, d representation dimensions, A actions. Built by
``analysis.fb_representation.compute_fb_representations`` (needs torch); this
record itself only needs NumPy, so dashboards can read it without torch.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from aeris.domain.artifacts import DiscreteState


@dataclass(frozen=True, eq=False)
class FBRepresentations:
    ARTIFACT_NAME: ClassVar[str] = "fb_representations_v1"

    run_id: str
    checkpoint: str
    gamma: float
    states: tuple[DiscreteState, ...]
    goals: tuple[DiscreteState, ...]
    backward: np.ndarray  # [S, d]       B(s)
    goal_conditions: np.ndarray  # [G, d]       z_g = B(goal_g)
    forward: np.ndarray  # [G, S, d, A] F(s, a, z_g)
    target_policy: np.ndarray  # [G, S, A]   Boltzmann policy of the target F for z_g
    state_density: np.ndarray  # [S]         rho: frequency of each state in the training data
    exact_density: np.ndarray  # [G, S, A, S] exact occupancy m(s, a, t) / rho(t); NaN where rho = 0

    def __post_init__(self) -> None:
        s, g = len(self.states), len(self.goals)
        d = self.backward.shape[1] if self.backward.ndim == 2 else -1
        a = self.forward.shape[-1] if self.forward.ndim == 4 else -1
        expected = {
            "backward": (s, d),
            "goal_conditions": (g, d),
            "forward": (g, s, d, a),
            "target_policy": (g, s, a),
            "state_density": (s,),
            "exact_density": (g, s, a, s),
        }
        for name, shape in expected.items():
            if getattr(self, name).shape != shape:
                raise ValueError(f"{name} has shape {getattr(self, name).shape}, expected {shape}")
        missing = set(self.goals) - set(self.states)
        if missing:
            raise ValueError(f"goals are not states: {sorted(missing)}")

    def goal_index(self, goal: DiscreteState) -> int:
        return self.goals.index(tuple(goal))

    def state_index(self, state: DiscreteState) -> int:
        return self.states.index(tuple(state))

    def q_values(self, goal: int) -> np.ndarray:
        """Learned Q(s, a) = F(s, a, z_g) · z_g, shape [S, A]."""
        return np.einsum("sda,d->sa", self.forward[goal], self.goal_conditions[goal])

    def predicted_density(self, goal: int) -> np.ndarray:
        """Learned occupancy density M(s, a, t) = F(s, a, z_g) · B(t), shape [S, A, S]."""
        return np.einsum("sda,td->sat", self.forward[goal], self.backward)

    def to_arrays(self) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        arrays = {
            name: getattr(self, name)
            for name in (
                "backward",
                "goal_conditions",
                "forward",
                "target_policy",
                "state_density",
                "exact_density",
            )
        }
        metadata = {
            "schema_version": 1,
            "run_id": self.run_id,
            "checkpoint": self.checkpoint,
            "gamma": self.gamma,
            "states": [list(s) for s in self.states],
            "goals": [list(g) for g in self.goals],
        }
        return arrays, metadata

    @classmethod
    def from_arrays(
        cls, arrays: Mapping[str, np.ndarray], metadata: Mapping[str, Any]
    ) -> FBRepresentations:
        if metadata.get("schema_version") != 1:
            raise ValueError(
                f"unsupported {cls.ARTIFACT_NAME} schema: {metadata.get('schema_version')!r}"
            )
        return cls(
            run_id=metadata["run_id"],
            checkpoint=metadata["checkpoint"],
            gamma=float(metadata["gamma"]),
            states=tuple(tuple(int(x) for x in s) for s in metadata["states"]),
            goals=tuple(tuple(int(x) for x in g) for g in metadata["goals"]),
            **{
                name: np.asarray(arrays[name])
                for name in (
                    "backward",
                    "goal_conditions",
                    "forward",
                    "target_policy",
                    "state_density",
                    "exact_density",
                )
            },
        )
