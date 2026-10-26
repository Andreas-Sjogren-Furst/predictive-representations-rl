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
    goal: tuple[int, int]
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
