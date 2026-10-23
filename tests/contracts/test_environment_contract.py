import pytest

from aeris.contracts import Environment
from aeris.domain import ResetResult, StepResult, Transition


class CounterWorld:
    """An environment with no Gymnasium or agent imports."""

    def __init__(self) -> None:
        self.x = 0
        self.closed = False

    def reset(self, *, seed: int | None = None) -> ResetResult:
        self.x = 0 if seed is None else seed % 3
        return ResetResult(observation=self.x, info={"seed": seed})

    def step(self, action: int) -> StepResult:
        self.x += action
        return StepResult(
            observation=self.x,
            reward=None,  # reward-free interaction is representable
            terminated=False,
            truncated=(self.x >= 4),
        )

    def close(self) -> None:
        self.closed = True


def test_environment_is_independent_and_can_build_a_transition():
    env = CounterWorld()
    assert isinstance(env, Environment)
    obs = env.reset(seed=3).observation
    result = env.step(4)
    transition = Transition(
        observation=obs,
        action=4,
        next_observation=result.observation,
        reward=result.reward,
        terminated=result.terminated,
        truncated=result.truncated,
    )
    assert transition.observation == 0
    assert transition.truncated is True
    assert transition.terminated is False
    assert transition.reward is None
    env.close()
    assert env.closed


@pytest.mark.parametrize("values, error", [
    ({"terminated": 0, "truncated": False}, TypeError),
    ({"terminated": False, "truncated": None}, TypeError),
    ({"terminated": True, "truncated": False, "reward": float("nan")}, ValueError),
])
def test_invalid_step_result_is_rejected(values, error):
    defaults = {"observation": 1, "reward": 0.0, "terminated": False, "truncated": False}
    defaults.update(values)
    with pytest.raises(error):
        StepResult(**defaults)
