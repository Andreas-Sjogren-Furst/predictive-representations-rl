import json

import numpy as np
import pytest

from aeris.data import RunDirectory
from aeris.domain.agent import ActionResult
from aeris.environments import FourRooms
from aeris.experiments import evaluate_goal_policy


class OraclePolicy:
    def __init__(self, env, goal):
        self.env, self.goal = env, goal
        self.calls = 0

    def act(self, request):
        self.calls += 1
        pos = self.env.position_of(request.observation)
        costs = [
            self.env.shortest_distance(self.env.next_position(pos, a), self.goal)
            for a in range(self.env.num_actions)
        ]
        return ActionResult(action=int(np.argmin(costs)))


def test_trajectory_capture_preserves_metrics_and_actual_transitions(tmp_path):
    goal = (9, 9)
    env = FourRooms(max_episode_steps=80)
    agent = OraclePolicy(env, goal)
    sink = RunDirectory(tmp_path)
    result = evaluate_goal_policy(
        agent,
        env,
        condition=None,
        is_goal=lambda obs: env.position_of(obs) == goal,
        episodes=5,
        max_steps=80,
        run_id="run",
        task_id="goal",
        checkpoint_id="ckpt",
        seed=90,
        on_trajectory=sink.write_trajectory,
        max_trajectories=2,
    )
    assert result.episode_count == 5
    assert result.metrics[0].value == 1.0
    traces = [
        json.loads(line) for line in (tmp_path / "trajectories.jsonl").read_text().splitlines()
    ]
    assert len(traces) == 2
    assert [x["seed"] for x in traces] == [90, 91]
    for trace in traces:
        assert trace["success"]
        assert len(trace["steps"]) <= 80
        assert trace["steps"][-1]["goal_reached"] or not trace["steps"]
        pos = env.position_of(trace["initial_observation"])
        for step in trace["steps"]:
            assert env.position_of(step["observation"]) == pos
            pos = env.next_position(pos, int(step["action"]))
            assert env.position_of(step["next_observation"]) == pos
        assert pos == goal


def test_no_trajectory_capture_leaves_no_file(tmp_path):
    env = FourRooms(max_episode_steps=2)
    agent = OraclePolicy(env, (9, 9))
    recorder = RunDirectory(tmp_path)
    evaluate_goal_policy(
        agent,
        env,
        condition=None,
        is_goal=lambda obs: False,
        episodes=2,
        max_steps=2,
        run_id="r",
        task_id="t",
        checkpoint_id="c",
        seed=2,
        on_trajectory=recorder.write_trajectory,
        max_trajectories=0,
    )
    assert not (tmp_path / "trajectories.jsonl").exists()


def test_serialization_rejects_nonfinite_without_corrupting_file(tmp_path):
    from aeris.domain.trajectory import EvaluationTrajectory, TrajectoryStep

    rec = EvaluationTrajectory(
        run_id="r",
        task_id="t",
        checkpoint_id="c",
        protocol_id="p",
        seed=1,
        episode_index=0,
        initial_observation=np.array([1.0]),
        steps=(
            TrajectoryStep(np.array([1.0]), 0, np.array([float("nan")]), None, False, False, False),
        ),
        success=False,
    )
    with pytest.raises(ValueError, match="finite"):
        RunDirectory(tmp_path).write_trajectory(rec)
    assert not (tmp_path / "trajectories.jsonl").exists()


def test_capture_does_not_change_metrics_or_action_counts():
    def evaluation(record):
        env = FourRooms(max_episode_steps=15)
        agent = OraclePolicy(env, (9, 9))
        traces = []
        record_ = evaluate_goal_policy(
            agent,
            env,
            condition=None,
            is_goal=lambda obs: env.position_of(obs) == (9, 9),
            episodes=5,
            max_steps=15,
            run_id="run",
            task_id="goal",
            checkpoint_id="ckpt",
            seed=90,
            on_trajectory=traces.append if record else None,
            max_trajectories=2,
        )
        return record_, agent.calls, traces

    no, calls1, _ = evaluation(False)
    yes, calls2, traces = evaluation(True)
    assert no == yes
    assert calls1 == calls2
    assert len(traces) == 2
