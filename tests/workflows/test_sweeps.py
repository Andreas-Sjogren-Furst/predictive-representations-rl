import stat
import subprocess
from pathlib import Path

import pytest
from typer.testing import CliRunner

from aeris.cli.main import app
from aeris.workflows import sweeps
from aeris.workflows.sweeps import SeedSweep, parse_seeds

runner = CliRunner()


def test_parse_seeds():
    assert parse_seeds("0-4") == (0, 1, 2, 3, 4)
    assert parse_seeds("3, 0-1,3") == (3, 0, 1)
    for bad in ("a", "4-2", "1-"):
        with pytest.raises(ValueError):
            parse_seeds(bad)


def test_each_seed_trains_then_runs_the_applicable_analyses():
    oracle = SeedSweep("fb", (0, 1), ("--updates", "10"), root=Path("runs"), python="py")
    steps = [command[3:5] for command in oracle.commands(1)]
    assert steps == [
        ["train", "four-rooms"],
        ["analysis", "four-rooms"],
        ["analysis", "zero-shot"],
        ["analysis", "representations"],
        ["analysis", "fb-checkpoint"],
    ]
    train = oracle.commands(1)[0]
    assert train[:3] == ["py", "-m", "aeris.cli.main"]
    assert ["--name", "fb", "--seed", "1", "--output", "runs/fb_seed1"] == train[5:11]
    assert train[-2:] == ["--updates", "10"]
    for args in (("--dataset", "random"), ("--dataset=random",)):
        assert not SeedSweep("fb", (0,), args).oracle_data  # no fb-checkpoint for random data


def test_lsf_script_has_one_array_element_per_seed(tmp_path):
    script = SeedSweep("fb_oracle", (0, 3, 7), python="/env/python").lsf_script(
        queue="hpc", walltime="2:00", memory="2GB", workdir=tmp_path
    )
    assert "#BSUB -J fb_oracle[1-3]" in script and "#BSUB -W 2:00" in script
    assert '#BSUB -R "rusage[mem=2GB]"' in script and f"cd {tmp_path}" in script
    assert "  2) # seed 3" in script and "--seed 7" in script
    assert script.count("analysis zero-shot") == 3


def test_local_runs_every_step_and_reports_failures(tmp_path):
    log = tmp_path / "calls.txt"
    fake = tmp_path / "python"
    fake.write_text(
        f'#!/bin/bash\necho "$@" >> {log}\n[[ "$*" == *"--seed 1"* ]] && exit 3\nexit 0\n'
    )
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    results = SeedSweep("fb", (0, 1), root=tmp_path, python=str(fake)).run_locally()
    assert results == {0: 0, 1: 3}
    calls = log.read_text().splitlines()
    assert len(calls) == 5 + 1  # seed 0: train + 4 analyses; seed 1 stops at its failed training


def test_submit_saves_the_script_and_calls_bsub(tmp_path, monkeypatch):
    seen = {}

    def fake_run(command, **kwargs):
        seen["command"] = command
        return subprocess.CompletedProcess(command, 0, stdout="Job <42> is submitted", stderr="")

    monkeypatch.setattr(sweeps.subprocess, "run", fake_run)
    sweep = SeedSweep("fb", (0,))
    path = tmp_path / "jobs" / "fb.lsf"
    script = sweep.lsf_script(workdir=tmp_path)
    assert sweep.submit_lsf(script, script_path=path) == "Job <42> is submitted"
    assert path.read_text() == script
    assert "bsub <" in seen["command"][-1] and "profile.lsf" in seen["command"][-1]
    assert (tmp_path / "logs" / "sweeps").is_dir()


def test_cli_dry_run_and_guards(tmp_path):
    result = runner.invoke(
        app,
        [
            "sweep",
            "four-rooms",
            "--name",
            "fb",
            "--seeds",
            "0-1",
            "--root",
            str(tmp_path),
            "--updates",
            "10",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Dry run" in result.output and "aeris train four-rooms" in result.output
    both = runner.invoke(
        app, ["sweep", "four-rooms", "--name", "fb", "--lsf", "--local", "--root", str(tmp_path)]
    )
    assert both.exit_code == 1 and "either --lsf or --local" in both.output
    (tmp_path / "fb_seed0").mkdir()
    exists = runner.invoke(app, ["sweep", "four-rooms", "--name", "fb", "--root", str(tmp_path)])
    assert exists.exit_code == 1 and "already exist" in exists.output
