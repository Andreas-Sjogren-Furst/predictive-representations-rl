import numpy as np
import pytest

from aeris.analysis.four_rooms import build_four_rooms_oracle, load_policy_actions
from aeris.analysis.tabular import compare_deterministic_policy, value_iteration
from aeris.environments import FourRooms


def test_value_iteration_simple_goal_ties():
    # 0 --(a0,a1)--> 1(goal); actions tie at state 0.
    nxt = np.array([[1, 1], [1, 1]])
    rewards = np.array([[1., 1.], [0., 0.]])
    optimal = value_iteration(nxt, rewards, np.array([False, True]), gamma=.9)
    assert optimal.values.tolist() == pytest.approx([1, 0])
    assert optimal.optimal_actions.tolist() == [[True, True], [False, False]]


def test_shortest_path_consistent_with_discounted_goal_value():
    env = FourRooms()
    goal = (9, 9)
    model = build_four_rooms_oracle(env, goal, gamma=.9)
    for i, pos in enumerate(model.positions):
        d = env.shortest_distance(pos, goal)
        if pos == goal:
            assert model.optimal.values[i] == pytest.approx(0)
        else:
            assert model.optimal.values[i] == pytest.approx(.9 ** (d - 1), abs=1e-9)
            for a in np.flatnonzero(model.optimal.optimal_actions[i]):
                new_d = env.shortest_distance(env.next_position(pos, int(a)), goal)
                assert new_d == d - 1


def test_optimal_policy_has_full_success_from_all_starts():
    env = FourRooms()
    model = build_four_rooms_oracle(env, (9, 9))
    policy = np.argmax(model.optimal.optimal_actions, axis=1).astype(np.int64)
    diag = compare_deterministic_policy(policy, model.next_state_indices, model.optimal,
       goal_index=model.goal_index, shortest_distances=model.shortest_distances, max_steps=100)
    assert diag.optimal_action_fraction == pytest.approx(1.)
    assert diag.rollout_success_fraction == pytest.approx(1.)
    assert diag.mean_action_regret == pytest.approx(0.)
    assert diag.mean_excess_steps_on_success == pytest.approx(0.)


def test_policy_loop_does_not_accidentally_count_as_success():
    env = FourRooms()
    model = build_four_rooms_oracle(env, (9, 9))
    policy = np.zeros(len(model.positions), dtype=np.int64)  # always up
    diag = compare_deterministic_policy(policy, model.next_state_indices, model.optimal,
       goal_index=model.goal_index, shortest_distances=model.shortest_distances, max_steps=100)
    assert diag.rollout_success_fraction == 0.0
    assert diag.optimal_action_fraction < 1.0
    assert diag.mean_excess_steps_on_success is None


def test_snapshot_alignment_with_shuffled_state_order(tmp_path):
    env = FourRooms()
    model = build_four_rooms_oracle(env, (9, 9))
    positions = np.array(model.positions)
    actions = np.argmax(model.optimal.optimal_actions, axis=1).astype(np.int64)
    order = np.random.default_rng(2).permutation(len(positions))
    snap = tmp_path / "policy.npz"
    np.savez_compressed(snap, positions=positions[order], actions=actions[order])
    aligned = load_policy_actions(snap, model.positions)
    diag = model.compare(aligned, max_steps=100)
    assert np.array_equal(aligned, actions)
    assert diag.optimal_action_fraction == 1.


def test_snapshot_missing_state_rejected(tmp_path):
    env = FourRooms()
    model = build_four_rooms_oracle(env, (9, 9))
    positions = np.array(model.positions)
    positions[-1] = positions[-2]
    snap = tmp_path / "bad.npz"
    np.savez_compressed(snap, positions=positions, actions=np.zeros(len(positions), dtype=int))
    with pytest.raises(ValueError, match="don't match"):
        load_policy_actions(snap, model.positions)


def test_bad_mdp_input_and_discount_rejected():
    with pytest.raises(ValueError, match="gamma"):
        value_iteration(np.array([[0]]), np.array([[0.]]), np.array([False]), gamma=1.0)
    with pytest.raises(ValueError, match="out-of-range"):
        value_iteration(np.array([[1]]), np.array([[0.]]), np.array([False]))
