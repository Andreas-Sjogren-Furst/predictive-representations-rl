import numpy as np
import pytest

from aeris.contracts import TabularModel, TransitionTable
from aeris.domain import TaskSpec
from aeris.environments import FourRooms, TabularDynamics, TabularMDP


def chain(n=3):
    """0 -> 1 -> ... -> n-1 (absorbing) with a second 'stay' action."""
    table = [[min(i + 1, n - 1), i] for i in range(n)]
    return TabularDynamics(tuple((i,) for i in range(n)), np.array(table))


def test_dynamics_are_validated_and_read_only():
    dynamics = chain()
    assert isinstance(dynamics, TransitionTable)
    assert (dynamics.num_states, dynamics.num_actions) == (3, 2)
    assert dynamics.index_of((2,)) == 2
    with pytest.raises(ValueError):
        dynamics.next_states[0, 0] = 0
    with pytest.raises(ValueError, match="valid integer state indices"):
        TabularDynamics(((0,), (1,)), np.array([[1], [2]]))
    with pytest.raises(ValueError, match="matching the states"):
        TabularDynamics(((0,),), np.array([[0], [0]]))
    with pytest.raises(ValueError, match="unique"):
        TabularDynamics(((0,), (0,)), np.array([[0], [1]]))
    with pytest.raises(ValueError, match="unknown state"):
        dynamics.index_of((7,))


def test_shortest_paths_mark_unreachable_states():
    dynamics = TabularDynamics(((0,), (1,), (2,)), np.array([[1], [1], [2]]))
    assert dynamics.shortest_path_lengths(1).tolist() == [1, 0, -1]


def test_goal_reaching_mdp_rewards_entering_the_goal_once():
    mdp = TabularMDP.goal_reaching(chain(), 2, gamma=0.9)
    assert mdp.rewards.tolist() == [[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]]
    assert mdp.terminals.tolist() == [False, False, True]


@pytest.mark.parametrize(
    "rewards, terminals, gamma, message",
    [
        (np.zeros((3, 2)), np.zeros(3, dtype=bool), 1.0, "gamma"),
        (np.zeros((3, 1)), np.zeros(3, dtype=bool), 0.9, "rewards"),
        (np.full((3, 2), np.inf), np.zeros(3, dtype=bool), 0.9, "rewards"),
        (np.zeros((3, 2)), np.zeros(3), 0.9, "terminals"),
    ],
)
def test_mdp_rejects_ill_formed_inputs(rewards, terminals, gamma, message):
    with pytest.raises(ValueError, match=message):
        TabularMDP(chain(), rewards, terminals, gamma)


def test_four_rooms_exposes_its_exact_model():
    env = FourRooms()
    assert isinstance(env, TabularModel)
    assert env.spec.describe() == "121-dim observations and 4 discrete actions"
    dynamics = env.tabular_dynamics()
    assert dynamics.states == env.positions
    assert dynamics.next_states[dynamics.index_of((4, 4)), 1] == dynamics.index_of((4, 4))  # wall


def test_goal_tasks_roundtrip_and_reject_walls():
    env = FourRooms()
    task = env.goal_task((9, 9))
    assert task.task_id == "reach_9_9"
    assert env.goal_from_task(task) == (9, 9)
    with pytest.raises(ValueError, match="traversable"):
        env.goal_task((5, 5))
    with pytest.raises(ValueError, match="goal_row"):
        env.goal_from_task(TaskSpec("other"))


def test_full_coverage_check():
    env = FourRooms()
    transitions = list(env.all_transitions())
    env.require_full_coverage(transitions)
    with pytest.raises(ValueError, match="duplicate or missing"):
        env.require_full_coverage(transitions[:-1])
    with pytest.raises(ValueError, match="duplicate or missing"):
        env.require_full_coverage([transitions[0], *transitions[1:-1], transitions[0]])
    wrong = transitions[0].__class__(transitions[0].observation, 0, transitions[5].next_observation)
    with pytest.raises(ValueError, match="incorrect transition"):
        env.require_full_coverage([wrong, *transitions[1:]])
