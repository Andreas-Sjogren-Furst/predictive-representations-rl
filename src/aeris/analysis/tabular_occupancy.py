"""Exact discounted state occupancies for small, known, finite MDPs.

Independent of neural networks, FB, Torch, and environment implementations.
Uses the convention G(s,a,x) = E[sum_{t>=0} gamma**t 1{s_t=x}
| s_0=s, a_0=a, a_{t>0}~pi(.|s_t)]. The t=0 term is the
CURRENT state; the chosen first action affects terms t>=1.
"""

from __future__ import annotations

import numpy as np


def exact_discounted_occupancy(
    next_states: np.ndarray, policy: np.ndarray, *, gamma: float
) -> np.ndarray:
    """Return occupancies [state, first_action, visited_state].

    next_states: deterministic model [S,A]; policy: action probabilities [S,A].
    No terminal handling: the policy keeps acting in the environment indefinitely.
    This intentionally matches FB's continuing-state visitation objective, NOT
    the finite-horizon, terminal-on-goal evaluation objective.
    """
    next_states = np.asarray(next_states)
    policy = np.asarray(policy, dtype=np.float64)
    if next_states.ndim != 2 or next_states.shape[0] == 0 or next_states.shape[1] < 2:
        raise ValueError("next_states must have shape [states, actions>=2]")
    if (
        next_states.dtype.kind not in "iu"
        or np.any(next_states < 0)
        or np.any(next_states >= len(next_states))
    ):
        raise ValueError("invalid next-state indices")
    if policy.shape != next_states.shape or not np.isfinite(policy).all() or np.any(policy < 0):
        raise ValueError("policy must contain finite nonnegative action probabilities")
    if not np.allclose(policy.sum(axis=1), 1.0, atol=1e-6, rtol=0):
        raise ValueError("policy probabilities must sum to one in each state")
    if not np.isfinite(gamma) or not 0 <= gamma < 1:
        raise ValueError("gamma must be finite and in [0, 1)")
    policy = policy / policy.sum(axis=1, keepdims=True)  # remove float32 rounding drift
    s, a = next_states.shape
    p_pi = np.zeros((s, s), dtype=np.float64)
    for j in range(a):
        np.add.at(p_pi, (np.arange(s), next_states[:, j]), policy[:, j])
    # V[s, x] = E[sum gamma^t * 1{s_t=x} | s_0=s, pi]
    v = np.linalg.solve(np.eye(s) - gamma * p_pi, np.eye(s))
    return np.eye(s)[:, None, :] + gamma * v[next_states]


def centered_pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    """Pearson correlation; None means at least one input lacks variation."""
    a = np.asarray(x, dtype=np.float64).ravel()
    b = np.asarray(y, dtype=np.float64).ravel()
    if not a.size or a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("arrays must be nonempty, equal-sized, and finite")
    a = a - a.mean()
    b = b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return None if denom <= 1e-14 else float(np.dot(a, b) / denom)
