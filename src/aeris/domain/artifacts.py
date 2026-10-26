"""Derived run artifacts that analyses consume instead of raw files."""

from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral

from .serialization import JsonRecord
from .spaces import DiscreteActionSpec

DiscreteState = tuple[int, ...]


def _as_int(name: str, value: object) -> int:
    # Accept NumPy integers from array-backed sources, but never bools or floats.
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must contain integers, got {value!r}")
    return int(value)


@dataclass(frozen=True, slots=True)
class PolicySnapshot(JsonRecord):
    """The deterministic action a policy chose in each enumerated discrete state.

    States are integer coordinate tuples (for Four Rooms, ``(row, column)``).
    Order is irrelevant: use :meth:`actions_for` to align with a model's states.
    """

    states: tuple[DiscreteState, ...]
    actions: tuple[int, ...]

    def __post_init__(self) -> None:
        states = tuple(tuple(_as_int("states", x) for x in s) for s in self.states)
        actions = tuple(_as_int("actions", a) for a in self.actions)
        if len(states) != len(actions):
            raise ValueError("policy snapshot needs exactly one action per state")
        if len(set(states)) != len(states):
            raise ValueError("policy snapshot contains duplicate states")
        if any(a < 0 for a in actions):
            raise ValueError("policy snapshot contains invalid actions")
        object.__setattr__(self, "states", states)
        object.__setattr__(self, "actions", actions)

    def require_action_space(self, space: DiscreteActionSpec) -> None:
        if not all(space.contains(a) for a in self.actions):
            raise ValueError("policy snapshot contains invalid actions")

    def actions_for(self, states: Sequence[DiscreteState]) -> tuple[int, ...]:
        """Actions in the order of ``states``, which must be exactly the recorded states."""
        mapping = dict(zip(self.states, self.actions, strict=True))
        if len(states) != len(mapping) or set(states) != set(mapping):
            raise ValueError("policy snapshot states don't match the environment")
        return tuple(mapping[tuple(s)] for s in states)
