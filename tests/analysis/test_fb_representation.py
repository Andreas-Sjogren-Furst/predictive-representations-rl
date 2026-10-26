"""The FB diagnostic needs only an agent, a tabular model and an oracle."""

import pytest

torch = pytest.importorskip("torch")
from aeris.agents.fb import DiscreteFBAgent, DiscreteFBConfig
from aeris.analysis.fb_representation import diagnose_fb_representation
from aeris.analysis.goal_policy import build_goal_oracle
from aeris.domain import PolicySnapshot
from aeris.environments import FourRooms
from aeris.experiments import record_policy_snapshot


def setup():
    torch.set_num_threads(1)
    env = FourRooms()
    agent = DiscreteFBAgent(
        DiscreteFBConfig(
            observation_dim=env.observation_dim,
            num_actions=env.num_actions,
            representation_dim=6,
            hidden_dim=10,
            seed=123,
        )
    )
    return env, agent, build_goal_oracle(env.tabular_dynamics(), (9, 9), gamma=agent.config.gamma)


def diagnose(env, agent, oracle, snapshot=None):
    return diagnose_fb_representation(
        agent,
        env,
        oracle,
        max_steps=15,
        run_id="r",
        checkpoint="checkpoint.pt",
        final_snapshot=snapshot,
    )


def test_untrained_agent_report_is_well_formed():
    env, agent, oracle = setup()
    report = diagnose(env, agent, oracle)
    assert report.gradient_updates == 0
    assert report.number_of_states == len(env.positions)
    assert 0 <= report.goal_action_agreement_with_exact_policy_improvement <= 1
    assert 0 <= report.backward_effective_rank <= 6
    assert sum(report.learned_action_counts_non_goal) == report.number_of_states - 1
    assert report.final_snapshot_matches_checkpoint is None


def test_snapshot_agreement_is_reported_not_raised():
    env, agent, oracle = setup()
    condition = agent.goal_condition(env.observation_for((9, 9)))
    own = record_policy_snapshot(agent, env.positions, env.observation_for, condition=condition)
    assert diagnose(env, agent, oracle, own).final_snapshot_matches_checkpoint is True
    flipped = PolicySnapshot(states=own.states, actions=[(a + 1) % 4 for a in own.actions])
    assert diagnose(env, agent, oracle, flipped).final_snapshot_matches_checkpoint is False
