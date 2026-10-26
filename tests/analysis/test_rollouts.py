import numpy as np
import pytest

from aeris.analysis.goal_policy import build_goal_oracle
from aeris.analysis.rollouts import (
    AT_GOAL,
    DETOUR,
    LATE,
    OPTIMAL,
    TRAPPED,
    policy_fates,
    rollout,
    successors,
)
from aeris.analysis.tabular import first_hit_steps
from aeris.environments import FourRooms, TabularDynamics

# A corridor 0 - 1 - 2 - 3; action 0 moves left, action 1 moves right (walls keep you in place).
LINE = TabularDynamics(
    states=((0,), (1,), (2,), (3,)), next_states=np.array([[0, 1], [0, 2], [1, 3], [2, 3]])
)
ENV = FourRooms()
DYN = ENV.tabular_dynamics()


def test_rollout_stops_at_the_goal():
    path = rollout(LINE, np.array([1, 1, 1, 1]), 0, goal=3)
    assert path.states == (0, 1, 2, 3) and path.actions == (1, 1, 1)
    assert path.goal_step == 3 and path.cycle == () and path.reached_goal


def test_rollout_closes_a_cycle_and_names_it():
    path = rollout(LINE, np.array([1, 1, 0, 1]), 0, goal=3)  # 1 -> 2 -> 1 -> ...
    assert path.states == (0, 1, 2, 1) and path.cycle_start == 1
    assert path.cycle == (1, 2) and not path.reached_goal and path.steps == 3


def test_a_wall_bump_is_a_one_state_cycle_and_the_goal_itself_takes_no_steps():
    stuck = rollout(LINE, np.array([0, 1, 1, 1]), 0, goal=3)
    assert stuck.states == (0, 0) and stuck.cycle == (0,) and stuck.cycle_start == 0
    assert rollout(LINE, np.zeros(4, dtype=int), 3, goal=3).goal_step == 0


def test_without_a_goal_every_rollout_ends_in_a_cycle():
    path = rollout(LINE, np.array([1, 1, 1, 1]), 0)
    assert path.states == (0, 1, 2, 3, 3) and path.cycle == (3,) and path.goal_step is None


def test_invalid_actions_are_rejected():
    with pytest.raises(ValueError, match="one integer action per state"):
        successors(LINE, np.array([0, 1]))
    with pytest.raises(ValueError, match="valid action"):
        successors(LINE, np.array([0, 1, 2, 0]))


def test_fates_group_starts_by_the_cycle_they_end_in():
    fates = policy_fates(LINE, np.array([1, 1, 0, 0]))  # 0 -> 1 <-> 2 <- 3
    assert fates.cycles == ((1, 2),)
    assert fates.cycle_of.tolist() == [0, 0, 0, 0]
    assert fates.transient.tolist() == [1, 0, 0, 1]
    assert fates.visits.tolist() == [0, 3, 3, 0]  # other starts that enter each state


def test_goal_outcomes_compare_with_the_shortest_path():
    fates = policy_fates(LINE, np.array([1, 1, 1, 1]), goal=3)
    assert fates.goal_outcomes(max_steps=5).tolist() == [OPTIMAL, OPTIMAL, OPTIMAL, AT_GOAL]
    assert fates.goal_outcomes(max_steps=2)[0] == OPTIMAL  # too slow, but nothing is faster
    looping = policy_fates(LINE, np.array([1, 1, 0, 1]), goal=3)
    assert looping.goal_outcomes(max_steps=5).tolist() == [TRAPPED, TRAPPED, TRAPPED, AT_GOAL]
    with pytest.raises(ValueError, match="need a goal"):
        policy_fates(LINE, np.array([1, 1, 1, 1])).goal_outcomes(max_steps=5)


def test_a_detour_is_late_when_it_misses_the_limit():
    # 4 states in a ring 0 -> 1 -> 2 -> 3 -> 0, plus a shortcut action from 0 straight to 3.
    ring = TabularDynamics(
        states=((0,), (1,), (2,), (3,)), next_states=np.array([[1, 3], [2, 1], [3, 2], [0, 3]])
    )
    fates = policy_fates(ring, np.array([0, 0, 0, 0]), goal=3)
    assert fates.goal_steps[0] == 3 and fates.shortest_steps is not None
    assert fates.shortest_steps[0] == 1
    assert fates.goal_outcomes(max_steps=10)[0] == DETOUR
    assert fates.goal_outcomes(max_steps=2)[0] == LATE


def test_fates_agree_with_first_hit_steps_on_four_rooms():
    rng = np.random.default_rng(0)
    goal = DYN.index_of((9, 9))
    oracle = build_goal_oracle(DYN, (9, 9), gamma=0.99)
    for actions in (oracle.optimal_policy(), *rng.integers(4, size=(20, DYN.num_states))):
        fates = policy_fates(DYN, actions, goal=goal)
        steps = first_hit_steps(actions, DYN, goal_index=goal, max_steps=DYN.num_states)
        assert np.array_equal(fates.goal_steps, steps)
    optimal = policy_fates(DYN, oracle.optimal_policy(), goal=goal)
    outcomes = optimal.goal_outcomes(max_steps=60)
    assert set(np.delete(outcomes, goal)) == {OPTIMAL}
