"""Backend-independent descriptions of what components exchange.

Environments declare the observations they emit and the actions they accept;
agents declare what they expect. Comparing the two before training replaces
ad-hoc dimension checks scattered through runners and analyses.
"""

from dataclasses import dataclass

from .serialization import JsonRecord
from .validation import require_int


@dataclass(frozen=True, slots=True)
class ObservationSpec:
    """A flat real-valued observation vector."""

    dim: int

    def __post_init__(self) -> None:
        require_int("observation dim", self.dim, minimum=1)


@dataclass(frozen=True, slots=True)
class DiscreteActionSpec:
    """Integer actions ``0, ..., n - 1``."""

    n: int

    def __post_init__(self) -> None:
        require_int("number of actions", self.n, minimum=1)

    def contains(self, action: object) -> bool:
        return isinstance(action, int) and not isinstance(action, bool) and 0 <= action < self.n


@dataclass(frozen=True, slots=True)
class SpaceSpec(JsonRecord):
    observation: ObservationSpec
    action: DiscreteActionSpec

    def require_compatible(self, provided: "SpaceSpec", *, consumer: str = "component") -> None:
        """Raise unless ``provided`` (e.g. an environment's spec) matches this one."""
        if provided != self:
            raise ValueError(f"{consumer} expects {self.describe()}, but got {provided.describe()}")

    def describe(self) -> str:
        return f"{self.observation.dim}-dim observations and {self.action.n} discrete actions"
