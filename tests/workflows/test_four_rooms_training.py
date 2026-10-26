import json

import pytest
from typer.testing import CliRunner

torch = pytest.importorskip("torch")
from aeris.agents.fb import DiscreteFBConfig
from aeris.agents.fb.checkpoint import load_agent
from aeris.cli.main import app
from aeris.workflows import four_rooms_training
from aeris.workflows.four_rooms import diagnose_fb_checkpoint
from aeris.workflows.four_rooms_training import FourRoomsFBTraining, train_four_rooms_fb

runner = CliRunner()
TINY = dict(updates=5, episodes=2, max_steps=10, trajectories=1)


def test_train_then_inspect_with_the_cli(tmp_path):
    out = tmp_path / "run"
    args = ["--updates", "5", "--episodes", "2", "--max-steps", "10", "--trajectories", "1"]
    trained = runner.invoke(app, ["train", "four-rooms", "--output", str(out), *args])
    assert trained.exit_code == 0, trained.output
    assert "Trained run: fb_four_rooms_seed_7" in trained.output
    for name in (
        "checkpoint.pt",
        "dataset.npz",
        "policy_initial.npz",
        "policy_final.npz",
        "training_metrics.jsonl",
        "evaluations.jsonl",
        "trajectories.jsonl",
    ):
        assert (out / name).is_file(), name

    assert runner.invoke(app, ["analysis", "four-rooms", str(out), "--no-plots"]).exit_code == 0
    diagnosis = runner.invoke(app, ["analysis", "fb-checkpoint", str(out), "--no-plots"])
    assert diagnosis.exit_code == 0, diagnosis.output
    assert "warning" not in diagnosis.output  # the saved final snapshot matches the checkpoint

    pytest.importorskip("matplotlib")
    plotted = runner.invoke(app, ["runs", "plot", str(out)])
    assert {"training.png", "policy_final.png", "trajectory_final.png"} <= {
        line.rsplit("/", 1)[-1] for line in plotted.output.splitlines()
    }


def test_manifest_records_the_full_agent_config(tmp_path):
    outcome = train_four_rooms_fb(FourRoomsFBTraining(output=tmp_path / "run", **TINY))
    settings = outcome.run.read_manifest().experiment.agent.settings
    assert DiscreteFBConfig(**settings) == load_agent(outcome.run.checkpoint_path).config


def test_refuses_a_nonempty_output_directory(tmp_path):
    (tmp_path / "existing").write_text("x")
    result = runner.invoke(app, ["train", "four-rooms", "--output", str(tmp_path)])
    assert result.exit_code == 1
    assert "must be empty" in result.output


def test_invalid_settings_are_rejected_before_anything_is_written(tmp_path):
    with pytest.raises(ValueError, match="updates"):
        FourRoomsFBTraining(output=tmp_path / "run", updates=0)
    with pytest.raises(ValueError, match="traversable"):
        train_four_rooms_fb(FourRoomsFBTraining(output=tmp_path / "run", goal=(5, 5), **TINY))
    assert not (tmp_path / "run").exists()


def test_a_failure_is_recorded_in_the_manifest(tmp_path, monkeypatch):
    def broken(*args, **kwargs):
        raise FloatingPointError("FB loss became nonfinite")

    monkeypatch.setattr(four_rooms_training, "train_updates", broken)
    with pytest.raises(FloatingPointError):
        train_four_rooms_fb(FourRoomsFBTraining(output=tmp_path / "run", **TINY))
    manifest = json.loads((tmp_path / "run" / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["error"] == "FloatingPointError: FB loss became nonfinite"


def test_diagnostic_refuses_a_checkpoint_from_another_configuration(tmp_path):
    from aeris.agents.fb import DiscreteFBAgent
    from aeris.agents.fb.checkpoint import save_checkpoint

    outcome = train_four_rooms_fb(FourRoomsFBTraining(output=tmp_path / "run", **TINY))
    other = DiscreteFBConfig(
        observation_dim=121, num_actions=4, representation_dim=32, hidden_dim=64, seed=7
    )
    save_checkpoint(outcome.run.checkpoint_path, DiscreteFBAgent(other))
    with pytest.raises(ValueError, match="hidden_dim .manifest 128, checkpoint 64."):
        diagnose_fb_checkpoint(outcome.run)
