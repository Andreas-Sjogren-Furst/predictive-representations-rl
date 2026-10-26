import numpy as np
import pytest

from aeris.domain import DiscreteActionSpec, ObservationSpec, PolicySnapshot, SpaceSpec


def spec(dim=121, n=4):
    return SpaceSpec(ObservationSpec(dim), DiscreteActionSpec(n))


def test_compatible_specs_pass_and_mismatches_explain_themselves():
    spec().require_compatible(spec())
    with pytest.raises(ValueError, match="expects 121-dim observations and 4 discrete actions"):
        spec().require_compatible(spec(n=5), consumer="agent")
    assert SpaceSpec.from_dict(spec().to_dict()) == spec()


@pytest.mark.parametrize("value", [0, -1, True, 2.0])
def test_space_sizes_must_be_positive_integers(value):
    with pytest.raises(ValueError):
        DiscreteActionSpec(value)


def test_action_membership():
    actions = DiscreteActionSpec(4)
    assert actions.contains(3)
    assert not actions.contains(4)
    assert not actions.contains(True)
    assert not actions.contains(1.0)


def test_policy_snapshot_aligns_to_any_state_order_and_accepts_numpy():
    snapshot = PolicySnapshot(states=np.array([[1, 1], [1, 2]]), actions=np.array([3, 0]))
    assert snapshot.states == ((1, 1), (1, 2))
    assert snapshot.actions_for([(1, 2), (1, 1)]) == (0, 3)
    assert PolicySnapshot.from_dict(snapshot.to_dict()) == snapshot


@pytest.mark.parametrize(
    "states, actions, message",
    [
        ([(1, 1)], [0, 1], "one action per state"),
        ([(1, 1), (1, 1)], [0, 1], "duplicate"),
        ([(1, 1)], [-1], "invalid actions"),
        ([(1, 1)], [0.5], "integers"),
    ],
)
def test_invalid_policy_snapshots_are_rejected(states, actions, message):
    with pytest.raises(ValueError, match=message):
        PolicySnapshot(states=states, actions=actions)


def test_snapshot_checks_states_and_action_space():
    snapshot = PolicySnapshot(states=[(1, 1)], actions=[5])
    with pytest.raises(ValueError, match="don't match"):
        snapshot.actions_for([(1, 1), (1, 2)])
    with pytest.raises(ValueError, match="invalid actions"):
        snapshot.require_action_space(DiscreteActionSpec(4))
