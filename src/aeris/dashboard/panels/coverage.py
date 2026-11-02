"""What the run's training data covers: start states, actions, and which tasks it can express."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import plotly.graph_objects as go

from aeris.dashboard.data import RunData
from aeris.dashboard.panels.base import STYLE, Choice, Option, Output, Panel
from aeris.dashboard.panels.grid import (
    ARROWS,
    COUNT_SCALE,
    ENV,
    INK,
    MUTED,
    add_goal,
    grid_figure,
    is_four_rooms,
    run_goal,
)
from aeris.domain.reports import ZeroShotSuiteReport

COLOURINGS = ("transitions", "actions covered")
_DYNAMICS = ENV.tabular_dynamics()


def state_action_counts(run: RunData) -> np.ndarray:
    """How often each (state, action) pair starts a transition in the training data, [S, A]."""
    assert run.dataset is not None
    counts = np.zeros((_DYNAMICS.num_states, _DYNAMICS.num_actions), dtype=int)
    for unit, action in zip(run.dataset.observations, run.dataset.actions, strict=True):
        counts[_DYNAMICS.index_of(divmod(int(unit), ENV.WIDTH)), int(action)] += 1
    return counts


def scenario_label(run: RunData) -> str:
    """The dataset as one phrase: 'oracle · room_top_left', 'random', ..."""
    origin = run.setting("dataset.origin")
    scenario = run.setting("dataset.scenario")
    return " · ".join(str(part) for part in (origin, scenario) if part is not None) or "unknown"


def _evidence(run: RunData) -> str:
    """Per family, how many suite tasks have at least one rewarded sample in the data."""
    report = run.report(ZeroShotSuiteReport)
    if not isinstance(report, ZeroShotSuiteReport):
        return ""
    families: dict[str, list[bool]] = {}
    for task in report.tasks:
        outcome = task.outcomes.get("dataset_rewards")
        if outcome is not None:
            families.setdefault(task.family, []).append(outcome.rewarded_samples > 0)
    parts = [
        f"{family.removeprefix('random_')} {sum(seen)}/{len(seen)}"
        for family, seen in families.items()
    ]
    return "tasks with rewarded evidence: " + ", ".join(parts)


@dataclass(frozen=True)
class DatasetCoverage(Panel):
    name = "dataset_coverage"
    title = "Dataset coverage"
    scope = "run"
    description = (
        "How often each cell starts a transition in the training data; arrows name the "
        "actions never taken there, outlined cells appear only as next states."
    )

    def options(self, runs: Sequence[RunData]) -> tuple[Option, ...]:
        return (Choice("colour", "Colour by", COLOURINGS, COLOURINGS[0]),)

    def missing(self, runs: Sequence[RunData]) -> str | None:
        run = runs[0]
        if not is_four_rooms(run):
            return "Dataset coverage is drawn for Four Rooms runs only."
        if run.dataset is None:
            return f"The run has no dataset file ({run.root}/dataset.npz)."
        return None

    def render(self, runs: Sequence[RunData], options: Mapping[str, Any]) -> Output:
        run = runs[0]
        assert run.dataset is not None
        counts = state_action_counts(run)
        starts = counts.sum(axis=1)
        taken = (counts > 0).sum(axis=1)
        states = _DYNAMICS.states
        by_actions = options["colour"] == COLOURINGS[1]
        values = taken if by_actions else starts
        cells = {s: float(v) for s, v, n in zip(states, values, starts, strict=True) if n > 0}
        top = _DYNAMICS.num_actions if by_actions else max(float(starts.max()), 1.0)
        fig = grid_figure(
            cells,
            zmin=0,
            zmax=top,
            colorscale=COUNT_SCALE,
            colorbar_title="actions" if by_actions else "transitions",
        )
        fig.data[-1].hovertemplate = (
            "row %{y}, col %{x}: %{z:.0f} " + ("actions" if by_actions else "transitions")
        ) + "<extra></extra>"
        gaps = [
            (s, "".join(ARROWS[a] for a in range(_DYNAMICS.num_actions) if counts[i, a] == 0))
            for i, s in enumerate(states)
            if 0 < taken[i] < _DYNAMICS.num_actions
        ]
        if gaps:
            fig.add_trace(
                go.Scatter(
                    x=[s[1] for s, _ in gaps],
                    y=[s[0] for s, _ in gaps],
                    mode="text",
                    text=[text for _, text in gaps],
                    # Readable on the colour beneath: light text on the darker half of the scale.
                    textfont={
                        "color": [
                            "white" if values[states.index(cell)] > top / 2 else INK
                            for cell, _ in gaps
                        ],
                        "size": 10,
                    },
                    name="actions never taken",
                    hovertemplate="%{text} never taken<extra></extra>",
                )
            )
        seen_next = {divmod(int(unit), ENV.WIDTH) for unit in run.dataset.next_observations}
        only_next = [s for i, s in enumerate(states) if starts[i] == 0 and s in seen_next]
        if only_next:
            fig.add_trace(
                go.Scatter(
                    x=[s[1] for s in only_next],
                    y=[s[0] for s in only_next],
                    mode="markers",
                    marker={
                        "symbol": "square-open",
                        "size": 22,
                        "color": MUTED,
                        "line": {"width": 2},
                    },
                    name="only a next state",
                    hovertemplate="row %{y}, col %{x}: only a next state<extra></extra>",
                )
            )
        goal = run_goal(run)
        if goal is not None:
            add_goal(fig, goal)
        covered = int((starts > 0).sum())
        pairs = int((counts > 0).sum())
        summary = (
            f"{len(run.dataset.actions):,} transitions · {covered}/{len(states)} start states · "
            f"{pairs}/{counts.size} state-action pairs"
        )
        lines = [summary, _evidence(run)]
        fig = STYLE.apply(
            fig,
            f"{self.title} · {scenario_label(run)}<br><sup>"
            + "<br>".join(line for line in lines if line)
            + "</sup>",
        )
        fig.update_layout(
            margin={**STYLE.margin, "t": 100}, legend={"orientation": "h", "y": -0.05}
        )
        return fig
