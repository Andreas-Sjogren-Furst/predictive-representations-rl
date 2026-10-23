from __future__ import annotations

from hashlib import sha256
import json

import numpy as np
import pytest

torch = pytest.importorskip('torch')
from aeris.agents.fb import DiscreteFBAgent, DiscreteFBConfig
from aeris.analysis.fb_checkpoint import analyze_fb_checkpoint
from aeris.environments import FourRooms


def make_run(tmp_path):
    torch.set_num_threads(1)
    env = FourRooms()
    transitions = tuple(env.all_transitions())
    dataset = tmp_path / 'dataset.npz'
    np.savez_compressed(
        dataset,
        observations=np.stack([t.observation for t in transitions]),
        actions=np.array([t.action for t in transitions], dtype=np.int64),
        next_observations=np.stack([t.next_observation for t in transitions]),
    )
    model = DiscreteFBAgent(DiscreteFBConfig(
        observation_dim=env.observation_dim, num_actions=env.num_actions,
        representation_dim=6, hidden_dim=10, seed=123,
    ))
    torch.save({'agent': model.state_dict()}, tmp_path / 'checkpoint.pt')
    manifest = {
        'run_id': 'test', 'status': 'completed', 'experiment': {
            'environment': {'name': 'four_rooms'},
            'dataset': {'settings': {'origin': 'oracle', 'sha256': sha256(dataset.read_bytes()).hexdigest()}},
            'evaluation': {
                'targets': [{'task': {'settings': {'goal_row': 9, 'goal_col': 9}}}],
                'protocol': {'settings': {'max_steps': 15}},
            },
        },
    }
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    return env, model, manifest


def test_checkpoint_diagnostic_does_not_retrain_or_require_matplotlib(tmp_path):
    _, _, _ = make_run(tmp_path)
    original = (tmp_path / 'checkpoint.pt').read_bytes()
    report = analyze_fb_checkpoint(tmp_path)
    assert report['gradient_updates'] == 0
    assert report['number_of_states'] > 40
    assert 0 <= report['goal_action_agreement_with_exact_policy_improvement'] <= 1
    assert 0 <= report['backward_effective_rank'] <= 6
    assert sum(report['learned_action_counts_non_goal']) == report['number_of_states'] - 1
    assert len(report['state_positions']) == report['number_of_states']
    assert len(report['state_action_improvement_agreement']) == report['number_of_states']
    assert report == json.loads((tmp_path / 'analysis/fb_checkpoint_v1.json').read_text())
    assert (tmp_path / 'checkpoint.pt').read_bytes() == original


def test_nonuniform_dataset_refused(tmp_path):
    _, _, manifest = make_run(tmp_path)
    manifest['experiment']['dataset']['settings']['origin'] = 'random'
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='oracle dataset'):
        analyze_fb_checkpoint(tmp_path)


def test_dataset_tampering_refused(tmp_path):
    _, _, _ = make_run(tmp_path)
    with (tmp_path / 'dataset.npz').open('ab') as f:
        f.write(b'tampered')
    with pytest.raises(ValueError, match='fingerprint'):
        analyze_fb_checkpoint(tmp_path)


def test_bad_state_action_coverage_refused_even_with_consistent_hash(tmp_path):
    _, _, manifest = make_run(tmp_path)
    dataset = tmp_path / 'dataset.npz'
    with np.load(dataset, allow_pickle=False) as data:
        observations, actions, nxt = data['observations'].copy(), data['actions'].copy(), data['next_observations'].copy()
    observations[1], actions[1], nxt[1] = observations[0].copy(), actions[0], nxt[0].copy()
    np.savez_compressed(dataset, observations=observations, actions=actions, next_observations=nxt)
    manifest['experiment']['dataset']['settings']['sha256'] = sha256(dataset.read_bytes()).hexdigest()
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='duplicate or missing'):
        analyze_fb_checkpoint(tmp_path)


def test_stale_snapshot_detected(tmp_path):
    env, _, _ = make_run(tmp_path)
    positions = np.asarray(env.positions)
    np.savez_compressed(tmp_path / 'policy_final.npz', positions=positions,
                        actions=np.full(len(positions), 10, dtype=np.int64))
    with pytest.raises(ValueError, match='invalid actions'):
        analyze_fb_checkpoint(tmp_path)
