import pytest

from aeris.agents.fb import FBCheckpointReport
from aeris.domain import PolicySummary
from aeris.environments import FourRooms


def test_render_alignment_from_recorded_data(tmp_path):
    pytest.importorskip("matplotlib")
    from aeris.visualization.fb_checkpoint import render_fb_action_alignment

    env = FourRooms()
    n = len(env.positions)
    summary = PolicySummary(
        optimal_action_fraction=1.0,
        mean_action_regret=0.0,
        maximum_action_regret=0.0,
        rollout_success_fraction=1.0,
        mean_excess_steps_on_success=0.0,
    )
    report = FBCheckpointReport(
        run_id="r",
        checkpoint="checkpoint.pt",
        goal=(9, 9),
        number_of_states=n,
        gradient_updates=0,
        gamma=0.99,
        policy_temperature=200.0,
        backward_effective_rank=1.0,
        backward_goal_norm=1.0,
        target_policy_mean_entropy_nats=1.0,
        target_policy_entropy_fraction_of_uniform=1.0,
        learned_action_spread_mean=0.0,
        learned_action_margin_mean=0.0,
        state_positions=env.positions,
        state_action_improvement_agreement=(True,) * n,
        learned_action_counts_non_goal=(n - 1, 0, 0, 0),
        exact_occupancy_correlation=None,
        goal_action_advantage_correlation=None,
        goal_action_agreement_with_exact_policy_improvement=1.0,
        first_hit_goal_performance=summary,
    )
    path = render_fb_action_alignment(report, tmp_path / "alignment.png")
    assert path.is_file() and path.stat().st_size > 1000
