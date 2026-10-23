"""Minimal working FB training example, independent of any environment package.

Usage: uv run --extra fb-torch python examples/discrete_fb_smoke.py
This is a smoke test, NOT reproduction of published maze results.
"""
from __future__ import annotations

import numpy as np
from aeris.agents.fb import DiscreteFBAgent, DiscreteFBConfig, FBTransitionSampler
from aeris.domain.agent import ActionRequest
from aeris.domain.experience import Transition


def main() -> None:
    n_states, n_actions = 6, 2
    eye = np.eye(n_states, dtype=np.float32)
    # Deterministic ring. Agent never sees these dynamics; sampler supplies data.
    dataset = []
    for state in range(n_states):
        for action, shift in enumerate((-1, 1)):
            dataset.append(Transition(observation=eye[state], action=action,
                                      next_observation=eye[(state + shift) % n_states]))
    batch_source = FBTransitionSampler(dataset, batch_size=32, seed=17)
    agent = DiscreteFBAgent(DiscreteFBConfig(observation_dim=n_states, num_actions=n_actions,
                                             representation_dim=16, hidden_dim=64,
                                             task_sampling="cauchy_ball", seed=17))
    for _ in range(30):
        metrics = agent.update(batch_source.next_batch())
    goal = agent.goal_condition(eye[3])
    action = agent.act(ActionRequest(observation=eye[0], condition=goal)).action
    print(f"FB updates: {agent.updates}; first-state action for goal 3: {action}")
    print({m.name: round(m.value, 5) for m in metrics})
    print("Smoke test only: no performance or paper-reproduction claim.")


if __name__ == "__main__":
    main()
