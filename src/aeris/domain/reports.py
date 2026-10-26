"""Typed analysis reports saved next to a run as derived artifacts.

Each report class names its artifact file, so stores can save and reload it
without knowing which analysis produced it. Constant fields document the
definitions a report was computed under; they are stored for readers of the
JSON file and fixed by the schema version.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import ClassVar

from .serialization import JsonRecord
from .task import TaskSpec
from .validation import require_identifier, require_int


class AnalysisReport(JsonRecord):
    """Base for reports; subclasses set ``ARTIFACT_NAME`` and ``schema_version``."""

    __slots__ = ()
    ARTIFACT_NAME: ClassVar[str]


@dataclass(frozen=True, slots=True, kw_only=True)
class PolicySummary:
    """Exact, model-based quality of one deterministic policy over non-goal start states."""

    optimal_action_fraction: float
    mean_action_regret: float
    maximum_action_regret: float
    rollout_success_fraction: float
    mean_excess_steps_on_success: float | None


@dataclass(frozen=True, slots=True, kw_only=True)
class SnapshotReport(PolicySummary):
    policy_snapshot: str
    success_gap_to_optimal: float


@dataclass(frozen=True, slots=True, kw_only=True)
class TabularPolicyReport(AnalysisReport):
    """Saved policy snapshots compared with the exact optimal goal-reaching policy."""

    ARTIFACT_NAME: ClassVar[str] = "tabular_policy_v1.json"

    schema_version: int = 1
    analysis: str = "four_rooms_tabular_policy_v1"
    run_id: str
    task_id: str | None
    goal: tuple[int, ...]
    reward_definition: str = "reward 1 on entering goal; goal terminal; 0 otherwise"
    pretraining_rewards_used: bool = False
    discount: float
    rollout_horizon: int
    state_weighting: str = "uniform over non-goal traversable states"
    tie_handling: str = "every action with Q*(s,a) >= max_a Q*(s,a)-1e-8 counts as optimal"
    optimal_policy: PolicySummary
    maximum_shortest_path_steps: int
    snapshots: Mapping[str, SnapshotReport]

    def __post_init__(self) -> None:
        require_identifier("run_id", self.run_id)
        require_int("rollout_horizon", self.rollout_horizon, minimum=1)
        if not self.snapshots:
            raise ValueError("a tabular policy report needs at least one snapshot")
        object.__setattr__(self, "snapshots", MappingProxyType(dict(self.snapshots)))


@dataclass(frozen=True, slots=True, kw_only=True)
class SuiteOutcome:
    """One task under one evidence protocol: the inferred policy's exact score."""

    normalized_score: float | None  # (V^pi - V^rand) / (V* - V^rand); None if V* = V^rand
    policy_value: float  # mean over start states
    goal_success: float | None  # goal tasks only: non-goal starts that first-hit the goal in time
    evidence_samples: int
    rewarded_samples: int  # evidence samples with nonzero reward; 0 means the task was invisible
    actions: tuple[int, ...]  # greedy action in each state, in the report's ``states`` order


@dataclass(frozen=True, slots=True, kw_only=True)
class SuiteTaskResult:
    task: TaskSpec
    family: str
    optimal_value: float  # mean V* over start states
    random_value: float  # mean V of the uniformly random policy
    outcomes: Mapping[str, SuiteOutcome]  # keyed by evidence protocol

    def __post_init__(self) -> None:
        object.__setattr__(self, "outcomes", MappingProxyType(dict(self.outcomes)))


@dataclass(frozen=True, slots=True, kw_only=True)
class ZeroShotSuiteReport(AnalysisReport):
    """Exact zero-shot scores of one checkpoint on every task of a task suite."""

    ARTIFACT_NAME: ClassVar[str] = "zero_shot_suite_v1.json"

    schema_version: int = 1
    analysis: str = "zero_shot_task_suite_v1"
    run_id: str
    checkpoint: str
    suite: str
    suite_description: str
    objective: str = (
        "discounted sum of state rewards, current state included, continuing policy; "
        "values averaged uniformly over start states"
    )
    score_definition: str = "(V^pi - V^random) / (V* - V^random): 0 = random policy, 1 = optimal"
    gamma: float
    max_steps: int  # horizon of the goal-success check
    evidence_source: str
    evidence_protocols: Mapping[str, str]  # name -> definition
    states: tuple[tuple[int, ...], ...]
    tasks: tuple[SuiteTaskResult, ...]
    mean_score: Mapping[str, Mapping[str, float | None]]  # family/"all" -> protocol -> mean score

    def __post_init__(self) -> None:
        require_identifier("run_id", self.run_id)
        require_int("max_steps", self.max_steps, minimum=1)
        if not self.tasks:
            raise ValueError("a suite report needs at least one task")
        protocols = set(self.evidence_protocols)
        for result in self.tasks:
            if set(result.outcomes) != protocols:
                raise ValueError("every task needs one outcome per evidence protocol")
            if any(len(o.actions) != len(self.states) for o in result.outcomes.values()):
                raise ValueError("outcome actions need one entry per state")
        for name in ("evidence_protocols", "mean_score"):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))

    def result(self, task_id: str) -> SuiteTaskResult:
        return next(r for r in self.tasks if r.task.task_id == task_id)
