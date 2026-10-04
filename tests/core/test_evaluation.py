import numpy as np

from predictive_representations_rl.core.backend import Policy
from predictive_representations_rl.core.evaluation import evaluate


class ConstantPolicy(Policy):
    def initial_state(self):
        return None

    def act(self, observation, state, *, deterministic=True):
        return np.zeros(2), state


class CountdownEnv:
    """Gymnasium-style env that gives reward 1 per step and ends after `episode_length` steps."""

    def __init__(self, episode_length):
        self.episode_length = episode_length
        self.steps = 0

    def reset(self, *, seed=None):
        self.steps = 0
        return np.zeros(4), {}

    def step(self, action):
        self.steps += 1
        truncated = self.steps >= self.episode_length
        return np.zeros(4), 1.0, False, truncated, {}


def test_evaluate_collects_returns_and_lengths():
    result = evaluate(ConstantPolicy(), CountdownEnv(episode_length=5), num_episodes=3)

    np.testing.assert_array_equal(result.episode_returns, [5.0, 5.0, 5.0])
    np.testing.assert_array_equal(result.episode_lengths, [5, 5, 5])
    assert result.mean_return == 5.0
    assert result.std_return == 0.0
    assert result.mean_length == 5.0
