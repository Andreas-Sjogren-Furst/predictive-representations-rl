"""Q-values of a fixed policy: exact, from the dataset alone, and learned by FB.

The objective is the suite's: the discounted sum of state rewards under a
continuing policy, counting the current state, so

    Q^pi(s, a) = r(s) + gamma * V^pi(next(s, a)).

``exact_q`` uses the true dynamics. ``dataset_q`` uses only the transitions
the dataset contains: a pair is known only when the data covers it and every
move the policy can make from there stays inside covered pairs; the rest is
NaN, since no learner could tell it from that data. ``PolicyEvaluation`` is
the saved artifact: each policy's table and learned forward vectors, from
which the learned Q of any reward follows as F^pi(s, a) · z_r.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from aeris.analysis.task_scores import policy_values
from aeris.domain.artifacts import DiscreteState
from aeris.environments.tabular import TabularDynamics

# Kinds of evaluated policy.
UNIFORM = "uniform"  # every action with probability 1/A
OPTIMAL = "optimal"  # uniform over the optimal actions of one task
GREEDY_FB = "fb greedy"  # the trained FB's own greedy policy for one task's inferred z
POLICY_KINDS = (UNIFORM, OPTIMAL, GREEDY_FB)


def exact_q(
    dynamics: TabularDynamics, policy: np.ndarray, rewards: np.ndarray, *, gamma: float
) -> np.ndarray:
    """Q^pi [S, A] from the true dynamics, for action probabilities ``policy`` [S, A]."""
    values = policy_values(dynamics, policy, rewards, gamma=gamma)
    return np.asarray(rewards, dtype=np.float64)[:, None] + gamma * values[dynamics.next_states]


def dataset_q(
    dynamics: TabularDynamics,
    policy: np.ndarray,
    rewards: np.ndarray,
    covered: np.ndarray,
    *,
    gamma: float,
) -> np.ndarray:
    """Q^pi [S, A] from the covered pairs only (``covered`` [S, A] bool); NaN where unknown.

    The dynamics are deterministic, so one sample of a pair gives its next state
    exactly; what the data cannot give is a pair it never contains.
    """
    s = dynamics.num_states
    rewards = np.asarray(rewards, dtype=np.float64)
    used = np.asarray(policy) > 0
    known = np.ones(s, dtype=bool)
    while True:  # drop states whose policy can leave the covered pairs, until none is left
        reaches = covered & known[dynamics.next_states]
        still = known & np.all(~used | reaches, axis=1)
        if np.array_equal(still, known):
            break
        known = still
    values = np.full(s, np.nan)
    index = np.flatnonzero(known)
    if index.size:
        position = {state: i for i, state in enumerate(index)}
        transition = np.zeros((index.size, index.size))
        for i, state in enumerate(index):
            for action in np.flatnonzero(used[state]):
                transition[i, position[dynamics.next_states[state, action]]] += policy[
                    state, action
                ]
        values[index] = np.linalg.solve(np.eye(index.size) - gamma * transition, rewards[index])
    q = rewards[:, None] + gamma * values[dynamics.next_states]
    q[~covered] = np.nan
    return q


def effective_horizon(
    dynamics: TabularDynamics,
    policy: np.ndarray,
    rewards: np.ndarray,
    learned: np.ndarray,
    *,
    gamma: float,
    longest: int = 300,
) -> tuple[int, float]:
    """The horizon H whose truncated Q^pi (the first H rewards only) best matches ``learned``
    [S, A], and that match's mean absolute error. TD learning with a slowly moving target
    propagates value one Bellman step per target refresh, so H counts the effective steps."""
    rewards = np.asarray(rewards, dtype=np.float64)
    s = dynamics.num_states
    transition = np.zeros((s, s))
    for action in range(dynamics.num_actions):
        np.add.at(transition, (np.arange(s), dynamics.next_states[:, action]), policy[:, action])
    values = np.zeros(s)  # V with the first H - 1 rewards
    best, best_error = 1, np.inf
    for horizon in range(1, longest + 1):
        q = rewards[:, None] + gamma * values[dynamics.next_states]
        error = float(np.abs(learned - q).mean())
        if error < best_error:
            best, best_error = horizon, error
        values = rewards + gamma * transition @ values
    return best, best_error


@dataclass(frozen=True)
class EvaluatedPolicy:
    label: str
    kind: str
    task: str | None  # the task it is optimal or greedy for; None for task-free policies

    def __post_init__(self) -> None:
        if self.kind not in POLICY_KINDS:
            raise ValueError(f"unknown policy kind {self.kind!r}")


@dataclass(frozen=True, eq=False)
class PolicyEvaluation:
    """Learned forward vectors of fixed policies, with the frozen B they were trained with.

    Shapes: P policies, S states, A actions, d dimensions, L logged losses.
    """

    ARTIFACT_NAME: ClassVar[str] = "policy_evaluation_v1"

    run_id: str
    checkpoint: str
    gamma: float
    updates: int  # training updates of each learned F^pi
    states: tuple[DiscreteState, ...]
    policies: tuple[EvaluatedPolicy, ...]
    tables: np.ndarray  # [P, S, A] action probabilities
    forward: np.ndarray  # [P, S, d, A] F^pi(s, a), or F(s, a, z) for FB's own policy
    backward: np.ndarray  # [S, d] the run's B, frozen during training
    state_density: np.ndarray  # [S] rho: each state's share of the dataset's start states
    covered: np.ndarray  # [S, A] bool: pairs the dataset contains
    losses: np.ndarray  # [P, L] training loss along training; NaN for untrained policies

    def __post_init__(self) -> None:
        p, s = len(self.policies), len(self.states)
        a = self.tables.shape[-1] if self.tables.ndim == 3 else -1
        d = self.backward.shape[1] if self.backward.ndim == 2 else -1
        expected = {
            "tables": (p, s, a),
            "forward": (p, s, d, a),
            "backward": (s, d),
            "state_density": (s,),
            "covered": (s, a),
        }
        for name, shape in expected.items():
            if getattr(self, name).shape != shape:
                raise ValueError(f"{name} has shape {getattr(self, name).shape}, expected {shape}")
        if self.losses.ndim != 2 or len(self.losses) != p:
            raise ValueError("losses must have one row per policy")
        if len({policy.label for policy in self.policies}) != p:
            raise ValueError("policy labels must be unique")

    def policy_index(self, label: str) -> int:
        return [policy.label for policy in self.policies].index(label)

    def reward_condition(self, rewards: np.ndarray) -> np.ndarray:
        """z_r = E_rho[r(s) B(s)]: the reward as FB reads it from the dataset's states."""
        return (self.state_density * np.asarray(rewards, dtype=np.float64)) @ self.backward

    def learned_q(self, policy: int, rewards: np.ndarray) -> np.ndarray:
        """Learned Q [S, A] = F^pi(s, a) · z_r."""
        return np.einsum("sda,d->sa", self.forward[policy], self.reward_condition(rewards))

    def to_arrays(self) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        arrays: dict[str, np.ndarray] = {
            "tables": self.tables.astype(np.float32),
            "forward": self.forward.astype(np.float32),
            "backward": self.backward.astype(np.float32),
            "state_density": self.state_density.astype(np.float64),
            "covered": self.covered.astype(bool),
            "losses": self.losses.astype(np.float64),
        }
        metadata: dict[str, Any] = {
            "schema_version": 1,
            "run_id": self.run_id,
            "checkpoint": self.checkpoint,
            "gamma": self.gamma,
            "updates": self.updates,
            "states": [list(state) for state in self.states],
            "policies": [{"label": p.label, "kind": p.kind, "task": p.task} for p in self.policies],
        }
        return arrays, metadata

    @classmethod
    def from_arrays(
        cls, arrays: Mapping[str, np.ndarray], metadata: Mapping[str, Any]
    ) -> PolicyEvaluation:
        if metadata.get("schema_version") != 1:
            raise ValueError(
                f"unsupported {cls.ARTIFACT_NAME} schema: {metadata.get('schema_version')!r}"
            )
        return cls(
            run_id=metadata["run_id"],
            checkpoint=metadata["checkpoint"],
            gamma=float(metadata["gamma"]),
            updates=int(metadata["updates"]),
            states=tuple(tuple(int(x) for x in state) for state in metadata["states"]),
            policies=tuple(
                EvaluatedPolicy(p["label"], p["kind"], p["task"]) for p in metadata["policies"]
            ),
            tables=np.asarray(arrays["tables"], dtype=np.float64),
            forward=np.asarray(arrays["forward"]),
            backward=np.asarray(arrays["backward"]),
            state_density=np.asarray(arrays["state_density"]),
            covered=np.asarray(arrays["covered"], dtype=bool),
            losses=np.asarray(arrays["losses"]),
        )
