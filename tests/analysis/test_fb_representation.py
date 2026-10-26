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


def test_saved_representations_agree_with_the_diagnostic():
    """F·B vs the exact density in the artifact gives the diagnostic's occupancy correlation."""
    import numpy as np

    from aeris.analysis.fb_representation import compute_fb_representations
    from aeris.analysis.tabular_occupancy import centered_pearson

    env, agent, oracle = setup()
    dynamics = env.tabular_dynamics()
    n = dynamics.num_states
    reps = compute_fb_representations(
        agent, env, dynamics, [(9, 9)], state_density=np.full(n, 1 / n), run_id="r", checkpoint="c"
    )
    report = diagnose(env, agent, oracle)
    non_goal = np.arange(n) != oracle.goal_index
    correlation = centered_pearson(
        reps.predicted_density(0)[non_goal], reps.exact_density[0][non_goal]
    )
    assert correlation == pytest.approx(report.exact_occupancy_correlation, abs=1e-4)
    assert reps.backward.shape == (n, agent.config.representation_dim)
