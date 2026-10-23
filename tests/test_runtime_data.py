import pytest

from aeris.domain import ActionRequest, ActionResult, Transition


def test_policy_request_supports_non_fb_and_recurrent_data():
    req = ActionRequest(observation={"pixels": [[1, 2]]}, policy_state=(1, 2))
    result = ActionResult(action=[-0.2, 0.5], next_policy_state=(2, 3))
    assert req.condition is None
    assert result.next_policy_state == (2, 3)


def test_reward_free_missing_end_flags_are_not_assumed_false():
    transition = Transition(0, 1, 2)
    assert transition.reward is None
    assert transition.terminated is None
    assert transition.truncated is None


def test_termination_distinct_from_time_limit():
    timeout = Transition(0, 1, 2, reward=1.0, terminated=False, truncated=True)
    assert not timeout.terminated
    assert timeout.truncated


@pytest.mark.parametrize("field,value", [("terminated", 1), ("truncated", "False")])
def test_transition_rejects_invalid_end_flag(field, value):
    with pytest.raises(TypeError):
        Transition(0, 1, 2, **{field: value})


def test_transition_rejects_non_finite_reward():
    with pytest.raises(ValueError):
        Transition(0, 1, 2, reward=float("inf"))
