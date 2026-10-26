"""Report schema for FB checkpoint diagnostics against an exact tabular model."""

from dataclasses import dataclass
from typing import ClassVar

from aeris.domain.reports import AnalysisReport, PolicySummary
from aeris.domain.validation import require_identifier, require_int


@dataclass(frozen=True, slots=True, kw_only=True)
class FBCheckpointReport(AnalysisReport):
    """Representation occupancy and action-ranking diagnostics for one goal."""

    ARTIFACT_NAME: ClassVar[str] = "fb_checkpoint_v1.json"

    schema_version: int = 1
    analysis: str = "fb_four_rooms_checkpoint_v1"
    run_id: str
    checkpoint: str
    goal: tuple[int, ...]
    number_of_states: int
    gradient_updates: int
    gamma: float
    policy_temperature: float
    dataset_assumption: str = "oracle full-coverage, uniform state density"
    policy_definition: str = "Boltzmann of target F(s, a, z)^T z with training temperature"
    occupancy_definition: str = (
        "discounted CURRENT-STATE visitation under continuing policy (no terminal state)"
    )
    evaluation_definition: str = (
        "first hitting goal within fixed horizon; terminal on entering goal"
    )
    backward_effective_rank: float
    backward_goal_norm: float
    target_policy_mean_entropy_nats: float
    target_policy_entropy_fraction_of_uniform: float
    learned_action_spread_mean: float
    learned_action_margin_mean: float
    state_positions: tuple[tuple[int, ...], ...]
    state_action_improvement_agreement: tuple[bool, ...]
    learned_action_counts_non_goal: tuple[int, ...]
    exact_occupancy_correlation: float | None
    goal_action_advantage_correlation: float | None
    goal_action_agreement_with_exact_policy_improvement: float
    first_hit_goal_performance: PolicySummary
    # Whether the greedy policy of this checkpoint equals the saved final policy
    # snapshot; None when the run has no such snapshot.
    final_snapshot_matches_checkpoint: bool | None = None

    def __post_init__(self) -> None:
        require_identifier("run_id", self.run_id)
        require_int("number_of_states", self.number_of_states, minimum=1)
        require_int("gradient_updates", self.gradient_updates)
        for name in ("state_positions", "state_action_improvement_agreement"):
            if len(getattr(self, name)) != self.number_of_states:
                raise ValueError(f"{name} must have one entry per state")
