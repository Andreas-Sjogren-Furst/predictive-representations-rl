import json

import numpy as np

from aeris.agents.fb import DiscreteFBAgent, DiscreteFBConfig, FBTransitionSampler
from aeris.contracts.agent import Agent, TrainableAgent
from aeris.data import RunDirectory
from aeris.domain.agent import ActionResult
from aeris.domain.results import MetricValue
from aeris.environments import FourRooms
from aeris.experiments import collect_transitions, evaluate_goal_policy, train_updates


class DummyBatches:
    def next_batch(self):
        return object()


class DummyAgent:
    def __init__(self, action=0):
        self.action = action
        self.calls = 0

    def act(self, request):
        assert request.deterministic
        return ActionResult(action=self.action)

    def update(self, batch):
        self.calls += 1
        return (
            MetricValue(
                name="dummy_loss", value=float(self.calls), unit="unitless", aggregation="instant"
            ),
        )


def test_runner_uses_only_capability_protocols(tmp_path):
    agent = DummyAgent()
    assert isinstance(agent, Agent)
    assert isinstance(agent, TrainableAgent)
    recorder = RunDirectory(tmp_path)
    summary = train_updates(
        agent,
        DummyBatches(),
        updates=3,
        start_step=8,
        run_id="test",
        recorder=recorder,
        log_every=2,
    )
    assert summary.update_count == 3
    rows = [
        json.loads(line) for line in (tmp_path / "training_metrics.jsonl").read_text().splitlines()
    ]
    assert [row["step"] for row in rows] == [10, 11]
    assert all(row["step_axis"] == "gradient_updates" for row in rows)


def test_collection_does_not_need_fb():
    env = FourRooms(max_episode_steps=3)

    def select(obs, rng):
        return int(rng.integers(4))

    one = collect_transitions(env, count=30, seed=99, choose_action=select)
    two = collect_transitions(
        FourRooms(max_episode_steps=3), count=30, seed=99, choose_action=select
    )
    assert one.completed_episodes == 10
    assert [t.action for t in one.transitions] == [t.action for t in two.transitions]
    assert all(
        np.array_equal(t.observation, u.observation)
        for t, u in zip(one.transitions, two.transitions, strict=True)
    )


def test_goal_evaluator_works_with_non_fb_agent():
    class GuidedPolicy:
        def __init__(self, env, goal):
            self.env, self.goal = env, goal

        def act(self, request):
            here = self.env.position_of(request.observation)
            distances = [
                self.env.shortest_distance(self.env.next_position(here, a), self.goal)
                for a in range(4)
            ]
            return ActionResult(action=int(np.argmin(distances)))

    env = FourRooms(max_episode_steps=40)
    goal = (9, 9)
    record = evaluate_goal_policy(
        GuidedPolicy(env, goal),
        env,
        condition=None,
        is_goal=lambda obs: env.position_of(obs) == goal,
        episodes=6,
        max_steps=40,
        run_id="test",
        checkpoint_id="oracle",
        task_id="reach_9_9",
        seed=123,
    )
    assert record.metrics[0].name == "success_rate"
    assert record.metrics[0].value == 1.0


def test_fb_agent_runs_end_to_end_without_environment_dependency():
    env = FourRooms(max_episode_steps=10)
    transitions = tuple(env.all_transitions())
    agent = DiscreteFBAgent(
        DiscreteFBConfig(
            observation_dim=env.observation_dim,
            num_actions=4,
            hidden_dim=32,
            representation_dim=8,
            seed=4,
        )
    )
    sampler = FBTransitionSampler(transitions, batch_size=8, seed=1)
    train_updates(agent, sampler, updates=3, run_id="smoke")
    before = agent.updates
    goal = env.observation_for((9, 9))
    record = evaluate_goal_policy(
        agent,
        env,
        condition=agent.goal_condition(goal),
        is_goal=lambda obs: np.array_equal(obs, goal),
        episodes=3,
        max_steps=5,
        run_id="smoke",
        task_id="goal",
        checkpoint_id="memory",
        seed=20,
    )
    assert agent.updates == before  # evaluation never trains
    assert 0 <= record.metrics[0].value <= 1
