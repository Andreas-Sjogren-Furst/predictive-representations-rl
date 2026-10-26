from __future__ import annotations

from hashlib import sha256

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from aeris.agents.fb import DiscreteFBAgent, DiscreteFBConfig, FBCheckpointReport
from aeris.agents.fb.checkpoint import save_checkpoint
from aeris.analysis.four_rooms.fb_checkpoint import analyze_fb_checkpoint
from aeris.data import RunDirectory, write_transitions_npz
from aeris.domain import PolicySnapshot
from aeris.environments import FourRooms


def make_run(tmp_path, four_rooms_run, *, origin="oracle", transitions=None):
    """An oracle-dataset run with an untrained FB checkpoint, as the training example saves it."""
    torch.set_num_threads(1)
    env = FourRooms()
    dataset = RunDirectory(tmp_path / "run").dataset_path
    fingerprint = write_transitions_npz(dataset, transitions or tuple(env.all_transitions()))
    run = four_rooms_run(max_steps=15, dataset_settings={"origin": origin, "sha256": fingerprint})
    model = DiscreteFBAgent(
        DiscreteFBConfig(
            observation_dim=env.observation_dim,
            num_actions=env.num_actions,
            representation_dim=6,
            hidden_dim=10,
            seed=123,
        )
    )
    save_checkpoint(run.checkpoint_path, model)
    return env, run


def test_checkpoint_diagnostic_does_not_retrain_or_require_matplotlib(tmp_path, four_rooms_run):
    _, run = make_run(tmp_path, four_rooms_run)
    original = run.checkpoint_path.read_bytes()
    report = analyze_fb_checkpoint(run)
    assert report.gradient_updates == 0
    assert report.number_of_states > 40
    assert 0 <= report.goal_action_agreement_with_exact_policy_improvement <= 1
    assert 0 <= report.backward_effective_rank <= 6
    assert sum(report.learned_action_counts_non_goal) == report.number_of_states - 1
    assert run.read_report(FBCheckpointReport) == report
    assert run.checkpoint_path.read_bytes() == original


def test_nonuniform_dataset_refused(tmp_path, four_rooms_run):
    _, run = make_run(tmp_path, four_rooms_run, origin="random")
    with pytest.raises(ValueError, match="oracle dataset"):
        analyze_fb_checkpoint(run)


def test_dataset_tampering_refused(tmp_path, four_rooms_run):
    _, run = make_run(tmp_path, four_rooms_run)
    with run.dataset_path.open("ab") as f:
        f.write(b"tampered")
    with pytest.raises(ValueError, match="fingerprint"):
        analyze_fb_checkpoint(run)


def test_bad_state_action_coverage_refused_even_with_consistent_hash(tmp_path, four_rooms_run):
    transitions = list(FourRooms().all_transitions())
    transitions[1] = transitions[0]
    _, run = make_run(tmp_path, four_rooms_run, transitions=tuple(transitions))
    with pytest.raises(ValueError, match="duplicate or missing"):
        analyze_fb_checkpoint(run)


def test_stale_snapshot_detected(tmp_path, four_rooms_run):
    env, run = make_run(tmp_path, four_rooms_run)
    run.write_policy_snapshot(
        "final", PolicySnapshot(states=env.positions, actions=np.full(len(env.positions), 10))
    )
    with pytest.raises(ValueError, match="invalid actions"):
        analyze_fb_checkpoint(run)


def test_dataset_fingerprint_is_sha256(tmp_path, four_rooms_run):
    _, run = make_run(tmp_path, four_rooms_run)
    expected = run.read_manifest().experiment.dataset.settings["sha256"]
    assert sha256(run.dataset_path.read_bytes()).hexdigest() == expected
