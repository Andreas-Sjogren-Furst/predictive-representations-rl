import json
import subprocess
import sys

import pytest

from aeris.agents.fb.reports import FBCheckpointReport
from aeris.domain import PolicySummary, SnapshotReport, TabularPolicyReport

SUMMARY = dict(
    optimal_action_fraction=0.5,
    mean_action_regret=0.1,
    maximum_action_regret=0.2,
    rollout_success_fraction=0.25,
    mean_excess_steps_on_success=None,
)


def tabular_report(**overrides):
    values = dict(
        run_id="run",
        task_id="reach_9_9",
        goal=(9, 9),
        discount=0.99,
        rollout_horizon=60,
        optimal_policy=PolicySummary(**SUMMARY),
        maximum_shortest_path_steps=16,
        snapshots={
            "final": SnapshotReport(
                **SUMMARY, policy_snapshot="policy_final.npz", success_gap_to_optimal=0.75
            )
        },
    )
    return TabularPolicyReport(**{**values, **overrides})


def test_tabular_report_keeps_the_v1_json_schema():
    data = json.loads(json.dumps(tabular_report().to_dict()))
    assert data["schema_version"] == 1
    assert data["analysis"] == "four_rooms_tabular_policy_v1"
    assert data["goal"] == [9, 9]
    assert data["snapshots"]["final"]["success_gap_to_optimal"] == 0.75
    assert TabularPolicyReport.from_dict(data) == tabular_report()
    assert TabularPolicyReport.ARTIFACT_NAME == "tabular_policy_v1.json"


def test_tabular_report_needs_a_snapshot():
    with pytest.raises(ValueError, match="at least one snapshot"):
        tabular_report(snapshots={})


def test_fb_report_requires_one_entry_per_state():
    values = dict(
        run_id="run",
        checkpoint="checkpoint.pt",
        goal=(9, 9),
        number_of_states=2,
        gradient_updates=0,
        gamma=0.99,
        policy_temperature=200.0,
        backward_effective_rank=1.0,
        backward_goal_norm=1.0,
        target_policy_mean_entropy_nats=1.0,
        target_policy_entropy_fraction_of_uniform=1.0,
        learned_action_spread_mean=0.0,
        learned_action_margin_mean=0.0,
        state_positions=((1, 1), (9, 9)),
        state_action_improvement_agreement=(True, False),
        learned_action_counts_non_goal=(1, 0, 0, 0),
        exact_occupancy_correlation=None,
        goal_action_advantage_correlation=0.1,
        goal_action_agreement_with_exact_policy_improvement=1.0,
        first_hit_goal_performance=PolicySummary(**SUMMARY),
    )
    report = FBCheckpointReport(**values)
    assert FBCheckpointReport.from_dict(json.loads(json.dumps(report.to_dict()))) == report
    with pytest.raises(ValueError, match="one entry per state"):
        FBCheckpointReport(**{**values, "state_action_improvement_agreement": (True,)})


def test_fb_report_does_not_import_torch():
    code = "import sys, aeris.agents.fb.reports; assert 'torch' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)
