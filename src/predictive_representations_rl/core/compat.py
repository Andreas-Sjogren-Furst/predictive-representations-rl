from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from predictive_representations_rl.algorithms.base import AlgorithmSpec
from predictive_representations_rl.core.config import ExperimentConfig
from predictive_representations_rl.envs.base import EnvSpec


@dataclass(frozen=True)
class Issue:
    # incompatible: the config can never run as written.
    # missing: the config is valid but a dataset, replay dir or runtime is not on disk yet.
    kind: Literal["incompatible", "missing"]
    message: str


def check(config: ExperimentConfig, env: EnvSpec, algo: AlgorithmSpec, root: Path) -> list[Issue]:
    """Everything that would stop `config` from running, found before anything is launched."""
    issues = [Issue("incompatible", msg) for msg in _compatibility(config, env, algo)]
    if not issues:
        issues += [Issue("missing", msg) for msg in _resources(config, env, algo, root)]
    return issues


def _compatibility(config: ExperimentConfig, env: EnvSpec, algo: AlgorithmSpec) -> list[str]:
    problems = []

    if config.mode not in algo.modes:
        problems.append(f"{algo.name} does not support {config.mode!r} learning; supported: {sorted(algo.modes)}")

    if env.action_type not in algo.action_spaces:
        problems.append(f"{algo.name} does not support {env.action_type} actions ({env.name})")

    if config.obs_type not in env.observation_types:
        problems.append(f"{env.name} has no {config.obs_type!r} observations; available: {sorted(env.observation_types)}")

    if config.obs_type not in algo.observation_types:
        problems.append(f"{algo.name} does not support {config.obs_type!r} observations")

    if algo.name not in env.native_names:
        problems.append(f"{env.name} has no name mapping for {algo.name}; add one to its EnvSpec.native_names")

    if algo.task_specific and config.task is None:
        problems.append(f"{algo.name} trains on a single task; set `task` to one of {list(env.tasks)}")
    elif not algo.task_specific and config.task is not None:
        problems.append(f"{algo.name} pretrains task-agnostically and is evaluated on every task; remove `task`")

    if config.task is not None and config.task not in env.tasks:
        problems.append(f"{env.name} has no task {config.task!r}; available: {list(env.tasks)}")

    requirement = algo.modes.get(config.mode)

    if requirement == "dataset":
        if config.dataset is None:
            problems.append(f"{algo.name} {config.mode} needs `dataset`; available for {env.name}: {sorted(env.datasets)}")
        elif config.dataset not in env.datasets:
            problems.append(f"{env.name} has no dataset {config.dataset!r}; available: {sorted(env.datasets)}")
        elif env.datasets[config.dataset].observation_type != config.obs_type:
            dataset_obs = env.datasets[config.dataset].observation_type
            problems.append(f"dataset {config.dataset!r} has {dataset_obs!r} observations, config asks for {config.obs_type!r}")
    elif config.dataset is not None:
        problems.append(f"{algo.name} {config.mode} does not use a dataset; remove `dataset`")

    if requirement == "replay" and config.replay_from is None:
        problems.append(f"{algo.name} {config.mode} trains on a previous run's replay; set `replay_from`")
    elif requirement != "replay" and config.replay_from is not None:
        problems.append(f"{algo.name} {config.mode} does not use `replay_from`; remove it")

    return problems


def _resources(config: ExperimentConfig, env: EnvSpec, algo: AlgorithmSpec, root: Path) -> list[str]:
    problems = list(algo.runtime.problems(root))

    if config.dataset is not None:
        dataset = env.datasets[config.dataset]
        for path in dataset.missing_files():
            problems.append(f"dataset file not found: {path}")
        if dataset.missing_files() and dataset.how_to_get:
            problems.append(f"to create {config.dataset!r}: {dataset.how_to_get}")

    if config.replay_from is not None and not (root / config.replay_from).expanduser().is_dir():
        problems.append(f"replay directory not found: {config.replay_from}")

    return problems
