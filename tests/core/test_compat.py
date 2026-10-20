from dataclasses import replace

import pytest

from predictive_representations_rl.algorithms.onestep_fb import ONESTEP_FB
from predictive_representations_rl.core.compat import check
from predictive_representations_rl.core.config import ExperimentConfig
from predictive_representations_rl.core.runtime import Venv
from predictive_representations_rl.envs.base import DatasetSpec
from predictive_representations_rl.envs.dmc import POINT_MASS_MAZE


@pytest.fixture
def root(tmp_path):
    """A fake project root with both algorithm venvs and an ExORL dataset on disk."""
    for venv in ("fb_venv", "task_venv"):
        (tmp_path / venv / "bin").mkdir(parents=True)
        (tmp_path / venv / "bin" / "python").touch()
    (tmp_path / "rnd.hdf5").touch()
    return tmp_path


@pytest.fixture
def env(root):
    dataset = DatasetSpec(name="exorl_rnd", observation_type="state", files=(str(root / "rnd.hdf5"),))
    native_names = {**POINT_MASS_MAZE.native_names, "task_algo": lambda task, obs_type: f"toy_{task}"}
    return replace(POINT_MASS_MAZE, datasets={"exorl_rnd": dataset}, native_names=native_names)


@pytest.fixture
def fb():
    return replace(ONESTEP_FB, runtime=Venv("fb_venv"))


@pytest.fixture
def task_algo():
    """A single-task algorithm that learns online from its own data or offline from another run's replay."""
    return replace(ONESTEP_FB, name="task_algo", modes={"online": "self_collected", "offline": "replay"},
                   task_specific=True, runtime=Venv("task_venv"))


def messages(issues):
    return " | ".join(issue.message for issue in issues)


def test_valid_configs_have_no_issues(root, env, fb, task_algo):
    fb_config = ExperimentConfig(env="point_mass_maze", algo="onestep_fb", mode="offline", dataset="exorl_rnd")
    task_config = ExperimentConfig(env="point_mass_maze", algo="task_algo", mode="online", task="reach_top_left")

    assert check(fb_config, env, fb, root) == []
    assert check(task_config, env, task_algo, root) == []


def test_unsupported_mode_is_incompatible(root, env, fb):
    config = ExperimentConfig(env="point_mass_maze", algo="onestep_fb", mode="online")

    issues = check(config, env, fb, root)

    assert issues and all(issue.kind == "incompatible" for issue in issues)
    assert "does not support 'online' learning" in messages(issues)


def test_discrete_env_rejects_continuous_only_algorithm(root, env, fb):
    discrete_env = replace(env, action_type="discrete")
    config = ExperimentConfig(env="point_mass_maze", algo="onestep_fb", mode="offline", dataset="exorl_rnd")

    assert "does not support discrete actions" in messages(check(config, discrete_env, fb, root))


def test_task_specific_algorithm_needs_a_valid_task(root, env, task_algo):
    no_task = ExperimentConfig(env="point_mass_maze", algo="task_algo", mode="online")
    bad_task = replace(no_task, task="reach_middle")

    assert "set `task`" in messages(check(no_task, env, task_algo, root))
    assert "has no task 'reach_middle'" in messages(check(bad_task, env, task_algo, root))


def test_task_agnostic_algorithm_rejects_a_task(root, env, fb):
    config = ExperimentConfig(env="point_mass_maze", algo="onestep_fb", mode="offline", dataset="exorl_rnd", task="reach_top_left")

    assert "remove `task`" in messages(check(config, env, fb, root))


def test_data_requirements_follow_the_mode(root, env, fb, task_algo):
    fb_without_dataset = ExperimentConfig(env="point_mass_maze", algo="onestep_fb", mode="offline")
    online_with_dataset = ExperimentConfig(env="point_mass_maze", algo="task_algo", mode="online", task="reach_top_left", dataset="exorl_rnd")
    offline_without_replay = ExperimentConfig(env="point_mass_maze", algo="task_algo", mode="offline", task="reach_top_left")

    assert "needs `dataset`" in messages(check(fb_without_dataset, env, fb, root))
    assert "does not use a dataset" in messages(check(online_with_dataset, env, task_algo, root))
    assert "set `replay_from`" in messages(check(offline_without_replay, env, task_algo, root))


def test_dataset_observation_type_must_match(root, env, fb):
    config = ExperimentConfig(env="point_mass_maze", algo="onestep_fb", mode="offline", dataset="exorl_rnd", obs_type="pixels")

    assert "has 'state' observations" in messages(check(config, env, fb, root))


def test_missing_resources_are_reported_separately(root, env, fb):
    (root / "rnd.hdf5").unlink()
    no_venv = replace(fb, runtime=Venv("not_there"))
    config = ExperimentConfig(env="point_mass_maze", algo="onestep_fb", mode="offline", dataset="exorl_rnd")

    issues = check(config, env, no_venv, root)

    assert issues and all(issue.kind == "missing" for issue in issues)
    assert "venv interpreter not found" in messages(issues)
    assert "dataset file not found" in messages(issues)


def test_registered_envs_map_every_registered_algorithm():
    from predictive_representations_rl.core.registry import ALGORITHMS, ENVS

    for env in ENVS.values():
        for name in env.native_names:
            assert name in ALGORITHMS
