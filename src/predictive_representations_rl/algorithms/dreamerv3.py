from __future__ import annotations

from predictive_representations_rl.algorithms.base import AlgorithmSpec
from predictive_representations_rl.core.runtime import Venv

DREAMERV3 = AlgorithmSpec(
    name="dreamerv3",
    description="DreamerV3 world model, online/offline fork (third_party/Offline_vs_Online_in_MBRL).",
    # online = active training; offline = passive training on another run's replay buffer.
    modes={"online": "self_collected", "offline": "replay"},
    action_spaces=frozenset({"continuous", "discrete"}),
    observation_types=frozenset({"state", "pixels"}),
    representations=("deter", "stoch", "model_state"),
    stateful_representation=True,
    task_specific=True,
    # Not set up yet on the HPC; the repo recommends its Singularity image instead.
    runtime=Venv("third_party/Offline_vs_Online_in_MBRL/.venv"),
)
