import json
import numpy as np
import pytest

from aeris.domain.agent import ActionResult
from aeris.environments import FourRooms
from aeris.visualization.four_rooms import save_policy_snapshot


class DownRight:
    def act(self, request):
        return ActionResult(action=3)


def test_policy_snapshot_uses_public_agent_contract(tmp_path):
    env = FourRooms()
    saved = save_policy_snapshot(DownRight(), env, None, tmp_path / "snapshot.npz")
    with np.load(saved, allow_pickle=False) as data:
        assert len(data["actions"]) == len(env.positions)
        assert (data["actions"] == 3).all()
        assert np.array_equal(data["positions"], np.asarray(env.positions))


def test_plot_from_saved_artifacts_only(tmp_path):
    pytest.importorskip("matplotlib")
    from aeris.visualization.four_rooms import render_policy_map, render_trajectory, render_training_metrics, render_evaluation_success
    env = FourRooms()
    save_policy_snapshot(DownRight(), env, None, tmp_path / "policy_initial.npz")
    plot = render_policy_map(tmp_path / "policy_initial.npz", (9, 9), tmp_path / "policy.png")
    assert plot.stat().st_size > 1000
    trace = {"checkpoint_id":"initial_in_memory", "episode_index":0, "success":False,
             "initial_observation":env.observation_for((1,1)).tolist(),
             "steps":[{"next_observation":env.observation_for((1,2)).tolist()}]}
    pic = render_trajectory(trace, (9,9), tmp_path / "trajectory.png")
    assert pic.stat().st_size > 1000
    with (tmp_path / "training_metrics.jsonl").open("w") as stream:
        for step in (10,20):
            stream.write(json.dumps({"step":step,"step_axis":"gradient_updates",
                                      "metrics":[{"name":"fb_loss","value":step/100}]}) + "\n")
    chart = render_training_metrics(tmp_path / "training_metrics.jsonl", tmp_path / "training.png")
    assert chart.stat().st_size > 1000
    (tmp_path / "evaluations.jsonl").write_text(json.dumps({
        "checkpoint_id":"checkpoint.pt", "episode_count":4,
        "metrics":[{"name":"success_rate","value":0.25}]
    }) + "\n")
    evaluation = render_evaluation_success(tmp_path / "evaluations.jsonl", tmp_path / "evaluation.png")
    assert evaluation.stat().st_size > 1000
