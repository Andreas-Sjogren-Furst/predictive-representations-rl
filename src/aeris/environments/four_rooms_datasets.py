"""Exact Four Rooms datasets: chosen state-action pairs, each a whole number of times.

An ``OracleDataset`` describes which states are start states (by room and by
cell), which actions are taken, how often each pair repeats, and which pairs
are removed at random. Every transition is the true one, so the only thing a
scenario changes is coverage: no sampling noise. Moves that leave the covered
states are kept, so a cell can appear as a next state without ever being a
start. ``PRESETS`` names the scenarios used in experiments; the full spec is
recorded in each run's manifest.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from aeris.domain.experience import Transition
from aeris.environments.four_rooms import DOORWAYS, ROOMS, FourRooms, Position, room_of

DOORWAY = "doorway"
AREAS = (*ROOMS, DOORWAY)  # every cell is in exactly one: a room or a doorway


@dataclass(frozen=True)
class OracleDataset:
    name: str
    description: str
    rooms: tuple[str, ...] = AREAS  # areas whose cells are start states
    add_cells: tuple[Position, ...] = ()  # extra start states outside those areas
    drop_cells: tuple[Position, ...] = ()  # never start states
    drop_actions: tuple[int, ...] = ()  # never taken anywhere
    drop_moves_touching: tuple[Position, ...] = ()  # no move that starts or ends in these cells
    weights: Mapping[str, int] = field(default_factory=dict)  # copies per area (default 1)
    holes: float = 0.0  # share of the remaining state-action pairs removed at random
    hole_seed: int = 0

    def __post_init__(self) -> None:
        unknown = set(self.rooms) - set(AREAS)
        if unknown or not self.rooms:
            raise ValueError(f"rooms must be a nonempty subset of {AREAS}")
        if set(self.weights) - set(AREAS) or any(
            not isinstance(w, int) or w < 1 for w in self.weights.values()
        ):
            raise ValueError(f"weights must map areas of {AREAS} to whole numbers >= 1")
        if not 0.0 <= self.holes < 1.0:
            raise ValueError("holes must be in [0, 1)")

    def pairs(self, env: FourRooms) -> list[tuple[Position, int, int]]:
        """(state, action, copies) of every covered pair, in grid then action order."""
        starts = [
            p
            for p in env.positions
            if (room_of(p) in self.rooms or p in self.add_cells) and p not in self.drop_cells
        ]
        touching = set(self.drop_moves_touching)
        found = [
            (p, a, self.weights.get(room_of(p), 1))
            for p in starts
            for a in range(env.num_actions)
            if a not in self.drop_actions
            and p not in touching
            and env.next_position(p, a) not in touching
        ]
        if self.holes:
            rng = np.random.default_rng(self.hole_seed)
            removed = set(
                rng.choice(len(found), size=round(self.holes * len(found)), replace=False).tolist()
            )
            found = [pair for i, pair in enumerate(found) if i not in removed]
        if not found:
            raise ValueError(f"dataset {self.name!r} covers no state-action pair")
        return found

    def transitions(self, env: FourRooms) -> tuple[Transition, ...]:
        """The dataset: every covered pair's true transition, repeated ``copies`` times."""
        return tuple(
            Transition(
                observation=env.observation_for(p),
                action=a,
                next_observation=env.observation_for(env.next_position(p, a)),
            )
            for p, a, copies in self.pairs(env)
            for _ in range(copies)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "rooms": list(self.rooms),
            "add_cells": [list(c) for c in self.add_cells],
            "drop_cells": [list(c) for c in self.drop_cells],
            "drop_actions": list(self.drop_actions),
            "drop_moves_touching": [list(c) for c in self.drop_moves_touching],
            "weights": dict(self.weights),
            "holes": self.holes,
            "hole_seed": self.hole_seed,
        }


_DOORWAYS = tuple(sorted(DOORWAYS))
_BUILT_IN = (
    OracleDataset("full", "every state-action pair once (the original oracle dataset)"),
    OracleDataset("room_top_left", "only the top-left room", rooms=("top-left",)),
    OracleDataset(
        "rooms_top",
        "the two top rooms and the doorway between them",
        rooms=("top-left", "top-right"),
        add_cells=((2, 5),),
    ),
    OracleDataset(
        "rooms_no_bottom_right",
        "every cell but the bottom-right room, where the default goal (9,9) is",
        rooms=("top-left", "top-right", "bottom-left", DOORWAY),
    ),
    OracleDataset("no_left", "every state, but ← is never taken", drop_actions=(2,)),
    OracleDataset(
        "no_doorway_moves",
        "no move into or out of any doorway: the rooms' connections are never seen",
        drop_moves_touching=_DOORWAYS,
    ),
    OracleDataset(
        "skew_top_left",
        "every pair, the top-left room's 10 times",
        weights={"top-left": 10},
    ),
    OracleDataset(
        "rare_doorways",
        "every pair 10 times, the doorways' once",
        weights={room: 10 for room in ROOMS},
    ),
    OracleDataset("no_goal", "every state but the default goal (9,9)", drop_cells=((9, 9),)),
    OracleDataset(
        "closed_doorway",
        "every state but the doorway (8,5) between the bottom rooms",
        drop_cells=((8, 5),),
    ),
    OracleDataset("holes_25", "a random 25% of the pairs removed", holes=0.25),
    OracleDataset("holes_50", "a random 50% of the pairs removed", holes=0.5),
)
PRESETS: dict[str, OracleDataset] = {spec.name: spec for spec in _BUILT_IN}


def oracle_dataset(name: str) -> OracleDataset:
    try:
        return PRESETS[name]
    except KeyError:
        raise ValueError(
            f"unknown dataset scenario {name!r}; available: {', '.join(PRESETS)}"
        ) from None
