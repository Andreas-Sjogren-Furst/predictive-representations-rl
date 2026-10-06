import json
import sys
from pathlib import Path

import numpy as np
import pytest

from predictive_representations_rl import runner
from predictive_representations_rl.algorithms.base import AlgorithmAdapter, AlgorithmSpec
from predictive_representations_rl.core import registry
from predictive_representations_rl.core.config import ExperimentConfig
from predictive_representations_rl.core.runtime import Command, Venv
from predictive_representations_rl.envs.base import EnvSpec
from predictive_representations_rl.probes.probe_set import ProbeSet

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FakeAdapter(AlgorithmAdapter):
    """Runs a tiny Python program that writes a checkpoint, or exits with the code in overrides['exit']."""

    spec = AlgorithmSpec(
        name="fake",
        description="test algorithm",
        modes={"online": "self_collected"},
        action_spaces=frozenset({"continuous"}),
        observation_types=frozenset({"state"}),
        representations=("z",),
        stateful_representation=False,
        task_specific=False,
        runtime=Venv("venv"),
    )

    def train_command(self, run):
        code = (
            "import os, pathlib, sys;"
            f"pathlib.Path({str(run.native_dir)!r}, 'model.ckpt').write_text('weights');"
            "print('training', os.environ['FAKE_FLAG']);"
            f"sys.exit({int(run.config.overrides.get('exit', 0))})"
        )
        return Command(args=("-c", code), cwd=run.root, env={"FAKE_FLAG": "on"})

    def find_checkpoint(self, run):
        path = run.native_dir / "model.ckpt"
        return path if path.exists() else None

    def extract_command(self, run, checkpoint, probes, out_dir):
        code = (
            "import json, sys, numpy as np, pathlib;"
            f"out = pathlib.Path({str(out_dir)!r});"
            f"obs = np.load({str(probes)!r})['observations'];"
            "np.savez(out / 'z.npz', values=obs[:, -1] * 2);"
            "(out / 'extract_info.json').write_text(json.dumps({'representations': {'z': list(obs[:, -1].shape)}}))"
        )
        return Command(args=("-c", code), cwd=run.root)

    def experience(self, run):
        return {"env_steps": run.config.budget.get("env_steps", 0)}


@pytest.fixture
def root(tmp_path, monkeypatch):
    (tmp_path / "pyproject.toml").touch()
    (tmp_path / "third_party").mkdir()
    (tmp_path / "venv" / "bin").mkdir(parents=True)
    # A wrapper, not a symlink: a symlinked venv python resolves to the base interpreter and loses site-packages.
    python = tmp_path / "venv" / "bin" / "python"
    python.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
    python.chmod(0o755)

    env = EnvSpec(
        name="toy", suite="test", tasks=("a",), action_type="continuous",
        observation_types=frozenset({"state"}), native_names={"fake": lambda task, obs: "toy-native"},
    )
    monkeypatch.setitem(registry.ENVS, "toy", env)
    monkeypatch.setitem(registry.ADAPTERS, "fake", FakeAdapter())
    monkeypatch.delenv(runner.RUNS_DIR_VARIABLE, raising=False)
    return tmp_path


def fake_config(**kwargs):
    return ExperimentConfig(env="toy", algo="fake", mode="online", budget={"env_steps": 10}, **kwargs)


def test_run_directory_layout(root):
    run = runner.resolve(fake_config(), seed=3, root=root)

    assert run.run_dir == root / "runs" / "toy" / "fake" / "online" / "seed_3"
    assert run.native_dir == run.run_dir / "native"
    assert run.native_env_name == "toy-native"


def test_runs_dir_can_point_elsewhere(root, tmp_path, monkeypatch):
    monkeypatch.setenv(runner.RUNS_DIR_VARIABLE, str(tmp_path / "scratch"))

    assert runner.resolve(fake_config(), seed=0, root=root).run_dir.is_relative_to(tmp_path / "scratch")


def test_run_locally_records_a_completed_run(root):
    run = runner.resolve(fake_config(), seed=0, root=root)

    record = runner.run_locally(run)

    on_disk = json.loads((run.run_dir / "run.json").read_text())
    assert json.loads(json.dumps(record, default=str)) == on_disk
    assert record["status"] == "completed" and record["exit_code"] == 0
    assert record["checkpoint"] == str(run.native_dir / "model.ckpt")
    assert record["experience"] == {"env_steps": 10}
    assert record["algorithm"]["data_requirement"] == "self_collected"
    assert record["command"]["env"] == {"FAKE_FLAG": "on"}
    assert "." in record["git"]
    assert "training on" in (run.run_dir / "train.log").read_text()


def test_failed_run_is_recorded(root):
    run = runner.resolve(fake_config(overrides={"exit": 3}), seed=0, root=root)

    record = runner.run_locally(run)

    assert record["status"] == "failed" and record["exit_code"] == 3


def test_finished_run_is_not_overwritten(root):
    run = runner.resolve(fake_config(), seed=0, root=root)
    runner.run_locally(run)

    with pytest.raises(runner.RunError, match="already has a completed run"):
        runner.run_locally(run)


def test_force_moves_the_previous_run_aside(root):
    run = runner.resolve(fake_config(), seed=0, root=root)
    runner.run_locally(run)

    runner.run_locally(run, force=True)

    replaced = list(run.run_dir.parent.glob("seed_0.replaced-*"))
    assert len(replaced) == 1 and (replaced[0] / "run.json").exists()
    assert json.loads((run.run_dir / "run.json").read_text())["status"] == "completed"


def test_submitted_run_can_be_started_by_its_job(root):
    run = runner.resolve(fake_config(), seed=0, root=root)
    run.run_dir.mkdir(parents=True)
    (run.run_dir / "run.json").write_text(json.dumps({"status": "submitted"}))

    assert runner.run_locally(run)["status"] == "completed"


def test_incompatible_config_is_rejected(root):
    with pytest.raises(runner.RunError, match="does not support 'offline'"):
        runner.resolve(ExperimentConfig(env="toy", algo="fake", mode="offline"), seed=0, root=root)


def test_lsf_script(root):
    run = runner.resolve(fake_config(resources={"queue": "gpuv100", "gpus": 1}), seed=2, root=root, config_path=root / "exp.yaml")

    script = runner._lsf_script(run)

    assert "#BSUB -q gpuv100" in script
    assert '#BSUB -gpu "num=1:mode=exclusive_process"' in script
    assert f"run {root / 'exp.yaml'} --seed 2" in script


def test_extract_writes_representations_and_records_them(root, monkeypatch):
    probe_set = ProbeSet(
        observations=np.arange(5 * 3 * 2, dtype=np.float32).reshape(5, 3, 2),
        actions=np.zeros((5, 3, 1), np.float32),
        metadata={"x": np.arange(5)},
    )
    run = runner.resolve(fake_config(), seed=0, root=root)
    probe_set.save(runner.probe_set_path(root, run.env))
    runner.run_locally(run)

    result = runner.extract(run)

    out = run.run_dir / "probes" / "default"
    np.testing.assert_array_equal(np.load(out / "z.npz")["values"], probe_set.final_observations * 2)
    np.testing.assert_array_equal(np.load(out / "metadata.npz")["x"], np.arange(5))
    assert result["representations"] == {"z": [5, 2]}
    assert json.loads((run.run_dir / "run.json").read_text())["extractions"]["default"]["checkpoint"].endswith("model.ckpt")


def test_analyze_adds_the_observation_baseline_and_records_results(root):
    rng = np.random.default_rng(0)
    observations = rng.uniform(-0.3, 0.3, (200, 3, 2)).astype(np.float32)
    probe_set = ProbeSet(
        observations=observations,
        actions=np.zeros((200, 3, 1), np.float32),
        metadata={"x": observations[:, -1, 0].astype(np.float64), "episode": np.repeat(np.arange(20), 10)},
    )
    run = runner.resolve(fake_config(), seed=0, root=root)
    probe_set.save(runner.probe_set_path(root, run.env))
    runner.run_locally(run)
    runner.extract(run)

    results = runner.analyze(run)

    assert set(results) == {"pca", "linear_probe"}
    summary = json.loads((run.run_dir / "analysis" / "default" / "linear_probe" / "summary.json").read_text())
    assert set(summary) == {"z", runner.BASELINE_REPRESENTATION, "skipped_factors"}
    assert summary["z"]["x"] > 0.99  # the fake extraction is 2 * final observation
    assert set(json.loads((run.run_dir / "run.json").read_text())["analyses"]["default"]) == {"pca", "linear_probe"}


def test_compare_cka_includes_the_baseline_and_rejects_stale_extractions(root):
    rng = np.random.default_rng(0)
    probe_set = ProbeSet(
        observations=rng.normal(size=(50, 2, 2)).astype(np.float32),
        actions=np.zeros((50, 2, 1), np.float32),
        metadata={"source_index": np.arange(50), "episode": np.repeat(np.arange(10), 5)},
    )
    run = runner.resolve(fake_config(), seed=0, root=root)
    probe_set.save(runner.probe_set_path(root, run.env))
    runner.run_locally(run)
    runner.extract(run)

    out = runner.compare_cka([run])

    header = (out / "cka.csv").read_text().splitlines()[0].split(",")
    assert header == ["representation", "fake/seed_0: z", runner.BASELINE_REPRESENTATION]
    assert (out / "cka.png").exists()

    # The fake extraction is 2 * the final observation, so CKA with the baseline is exactly 1.
    assert float((out / "cka.csv").read_text().splitlines()[1].split(",")[2]) == pytest.approx(1.0)

    ProbeSet(probe_set.observations, probe_set.actions, {**probe_set.metadata, "source_index": np.arange(50) + 1}).save(
        runner.probe_set_path(root, run.env)
    )
    with pytest.raises(runner.RunError, match="different version"):
        runner.compare_cka([run])


def test_analyze_needs_an_extraction(root):
    run = runner.resolve(fake_config(), seed=0, root=root)
    runner.run_locally(run)

    with pytest.raises(runner.RunError, match="prl extract"):
        runner.analyze(run)


def test_extract_needs_a_probe_set(root):
    run = runner.resolve(fake_config(), seed=0, root=root)
    runner.run_locally(run)

    with pytest.raises(runner.RunError, match="prl probes toy"):
        runner.extract(run)


def test_onestep_fb_command():
    config = ExperimentConfig(
        env="point_mass_maze", algo="onestep_fb", mode="offline", dataset="exorl_rnd",
        budget={"train_steps": 2000}, overrides={"agent.latent_dim": 50},
    )
    run = runner.resolve(config, seed=1, root=PROJECT_ROOT, allow_missing=True)

    argv, env, cwd = runner.command_line(run)

    assert argv[0].endswith("third_party/onestep-fb/.venv/bin/python") and argv[1] == "main.py"
    assert "--env_name=exorl-rnd-point_mass_maze" in argv
    assert f"--save_dir={run.native_dir}" in argv
    assert "--train_steps=2000" in argv and "--save_interval=2000" in argv
    assert "--agent.latent_dim=50" in argv
    assert cwd == PROJECT_ROOT / "third_party" / "onestep-fb"


def test_dreamerv3_command():
    config = ExperimentConfig(
        env="point_mass_maze", algo="dreamerv3", mode="online", task="reach_top_left",
        budget={"env_steps": 1500}, overrides={"run.train_ratio": 64},
    )
    run = runner.resolve(config, seed=0, root=PROJECT_ROOT, allow_missing=True)

    argv, env, cwd = runner.command_line(run)
    flags = dict(zip(argv[2::2], argv[3::2]))

    assert argv[1] == "dreamerv3/train.py"
    assert flags["--configs"] == "dmc_proprio"
    assert flags["--task"] == "dmc_point_mass_maze_reach_top_left"
    assert flags["--logdir"] == str(run.native_dir)
    assert flags["--run.steps"] == "1500" and flags["--run.train_ratio"] == "64"


def test_dreamerv3_extract_posterior_option():
    base = ExperimentConfig(env="point_mass_maze", algo="dreamerv3", mode="online", task="reach_top_left")
    paths = (Path("ckpt"), Path("probes.npz"), Path("out"))

    def posterior_flag(config):
        run = runner.resolve(config, seed=0, root=PROJECT_ROOT, allow_missing=True)
        args = run.adapter.extract_command(run, *paths).args
        return args[args.index("--posterior") + 1]

    assert posterior_flag(base) == "sample"
    assert posterior_flag(ExperimentConfig(**{**base.__dict__, "extract": {"posterior": "mode"}})) == "mode"

    for bad in ({"posterior": "mean"}, {"temperature": 1}):
        run = runner.resolve(ExperimentConfig(**{**base.__dict__, "extract": bad}), seed=0, root=PROJECT_ROOT, allow_missing=True)
        with pytest.raises(ValueError):
            run.adapter.extract_command(run, *paths)


def test_onestep_fb_rejects_extract_options():
    config = ExperimentConfig(env="point_mass_maze", algo="onestep_fb", mode="offline", dataset="exorl_rnd", extract={"posterior": "mode"})
    run = runner.resolve(config, seed=0, root=PROJECT_ROOT, allow_missing=True)

    with pytest.raises(ValueError, match="takes no options"):
        run.adapter.extract_command(run, Path("ckpt"), Path("probes.npz"), Path("out"))
