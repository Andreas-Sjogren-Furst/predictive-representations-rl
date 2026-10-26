"""Exact discounted state occupancies for small, known, finite MDPs.

Independent of neural networks, FB and Torch; operates on validated TabularDynamics.
Uses the convention G(s,a,x) = E[sum_{t>=0} gamma**t 1{s_t=x}
| s_0=s, a_0=a, a_{t>0}~pi(.|s_t)]. The t=0 term is the
CURRENT state; the chosen first action affects terms t>=1.
"""

from __future__ import annotations

import numpy as np

from aeris.environments.tabular import TabularDynamics


def exact_discounted_occupancy(
    dynamics: TabularDynamics, policy: np.ndarray, *, gamma: float
) -> np.ndarray:
    """Return occupancies [state, first_action, visited_state].

    ``policy`` holds action probabilities [S, A]. No terminal handling: the
    policy keeps acting in the environment indefinitely. This intentionally
    matches FB's continuing-state visitation objective, NOT the finite-horizon,
    terminal-on-goal evaluation objective.
    """
    next_states = dynamics.next_states
    policy = np.asarray(policy, dtype=np.float64)
    policy = policy / policy.sum(axis=1, keepdims=True)  # remove float32 rounding drift
    s, a = next_states.shape
    p_pi = np.zeros((s, s), dtype=np.float64)
    for j in range(a):
        np.add.at(p_pi, (np.arange(s), next_states[:, j]), policy[:, j])
    # V[s, x] = E[sum gamma^t * 1{s_t=x} | s_0=s, pi]
    v = np.linalg.solve(np.eye(s) - gamma * p_pi, np.eye(s))
    return np.eye(s)[:, None, :] + gamma * v[next_states]


def centered_pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    """Pearson correlation of two equal-size arrays; None if either lacks variation."""
    a = np.asarray(x, dtype=np.float64).ravel()
    b = np.asarray(y, dtype=np.float64).ravel()
    a = a - a.mean()
    b = b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return None if denom <= 1e-14 else float(np.dot(a, b) / denom)
