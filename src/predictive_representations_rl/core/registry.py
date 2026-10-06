from __future__ import annotations

from predictive_representations_rl.algorithms.base import AlgorithmAdapter, AlgorithmSpec
from predictive_representations_rl.algorithms.dreamerv3 import DreamerV3Adapter
from predictive_representations_rl.algorithms.onestep_fb import OneStepFBAdapter
from predictive_representations_rl.envs.base import EnvSpec
from predictive_representations_rl.envs.dmc import POINT_MASS_MAZE

# To add an environment or algorithm: define it in envs/ or algorithms/ and list it here.
ENVS: dict[str, EnvSpec] = {spec.name: spec for spec in (POINT_MASS_MAZE,)}
ADAPTERS: dict[str, AlgorithmAdapter] = {adapter.spec.name: adapter for adapter in (OneStepFBAdapter(), DreamerV3Adapter())}
ALGORITHMS: dict[str, AlgorithmSpec] = {name: adapter.spec for name, adapter in ADAPTERS.items()}


def get_env(name: str) -> EnvSpec:
    if name not in ENVS:
        raise KeyError(f"Unknown environment {name!r}. Available: {', '.join(sorted(ENVS))}")
    return ENVS[name]


def get_adapter(name: str) -> AlgorithmAdapter:
    if name not in ADAPTERS:
        raise KeyError(f"Unknown algorithm {name!r}. Available: {', '.join(sorted(ADAPTERS))}")
    return ADAPTERS[name]


def get_algorithm(name: str) -> AlgorithmSpec:
    return get_adapter(name).spec
