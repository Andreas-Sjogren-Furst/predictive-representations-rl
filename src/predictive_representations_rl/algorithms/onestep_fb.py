from __future__ import annotations

from predictive_representations_rl.algorithms.base import AlgorithmSpec
from predictive_representations_rl.core.runtime import Venv

ONESTEP_FB = AlgorithmSpec(
    name="onestep_fb",
    description="One-step Forward-Backward representations (third_party/onestep-fb).",
    modes={"offline": "dataset"},
    action_spaces=frozenset({"continuous"}),
    observation_types=frozenset({"state", "pixels"}),
    representations=("forward", "backward", "latent"),
    stateful_representation=False,
    task_specific=False,
    runtime=Venv("third_party/onestep-fb/.venv"),
)
