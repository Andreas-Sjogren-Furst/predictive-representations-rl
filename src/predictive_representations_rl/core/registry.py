from __future__ import annotations

from predictive_representations_rl.algorithms.base import AlgorithmSpec
from predictive_representations_rl.algorithms.dreamerv3 import DREAMERV3
from predictive_representations_rl.algorithms.onestep_fb import ONESTEP_FB
from predictive_representations_rl.envs.base import EnvSpec
from predictive_representations_rl.envs.dmc import POINT_MASS_MAZE

# To add an environment or algorithm: define its spec in envs/ or algorithms/ and list it here.
ENVS: dict[str, EnvSpec] = {spec.name: spec for spec in (POINT_MASS_MAZE,)}
ALGORITHMS: dict[str, AlgorithmSpec] = {spec.name: spec for spec in (ONESTEP_FB, DREAMERV3)}


def get_env(name: str) -> EnvSpec:
    if name not in ENVS:
        raise KeyError(f"Unknown environment {name!r}. Available: {', '.join(sorted(ENVS))}")
    return ENVS[name]


def get_algorithm(name: str) -> AlgorithmSpec:
    if name not in ALGORITHMS:
        raise KeyError(f"Unknown algorithm {name!r}. Available: {', '.join(sorted(ALGORITHMS))}")
    return ALGORITHMS[name]
