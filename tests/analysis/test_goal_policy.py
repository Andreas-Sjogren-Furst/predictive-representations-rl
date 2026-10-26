"""Pure analysis: typed inputs in, typed report out, no run directory involved."""

import numpy as np
import pytest

from aeris.analysis.goal_policy import analyze_policy_snapshots, build_goal_oracle
from aeris.domain import PolicySnapshot
from aeris.environments import FourRooms


def oracle():
    return build_goal_oracle(FourRooms().tabular_dynamics(), (9, 9), gamma=0.99)


def test_optimal_snapshot_scores_perfectly_and_others_do_not():
    model = oracle()
    optimal = PolicySnapshot(states=model.states, actions=model.optimal_policy())
    always_up = PolicySnapshot(states=model.states, actions=np.zeros(len(model.states), dtype=int))
    analysis = analyze_policy_snapshots(
        model,
        {"final": optimal, "initial": always_up},
        max_steps=60,
        run_id="run",
        task_id="reach_9_9",
        snapshot_names={"final": "policy_final.npz", "initial": "policy_initial.npz"},
    )
    final, initial = analysis.report.snapshots["final"], analysis.report.snapshots["initial"]
    assert final.optimal_action_fraction == 1.0
    assert final.success_gap_to_optimal == pytest.approx(0)
    assert initial.rollout_success_fraction < final.rollout_success_fraction
    assert analysis.report.goal == (9, 9)
    assert analysis.report.discount == 0.99
    assert np.array_equal(analysis.snapshots["final"].actions, model.optimal_policy())


def test_snapshot_order_does_not_matter():
    model = oracle()
    order = np.random.default_rng(0).permutation(len(model.states))
    shuffled = PolicySnapshot(
        states=np.asarray(model.states)[order], actions=model.optimal_policy()[order]
    )
    analysis = analyze_policy_snapshots(
        model,
        {"final": shuffled},
        max_steps=60,
        run_id="r",
        task_id="t",
        snapshot_names={"final": "policy_final.npz"},
    )
    assert analysis.report.snapshots["final"].optimal_action_fraction == 1.0
