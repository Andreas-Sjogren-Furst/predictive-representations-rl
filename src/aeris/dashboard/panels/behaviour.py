"""Exact behaviour of zero-shot policies: replay one start, or map every start's fate.

Both panels read the greedy action in every cell that ``aeris analysis zero-shot``
saved for each suite task and evidence protocol, and roll it out on the exact
Four Rooms dynamics (``analysis.rollouts``): no checkpoint, no sampling, any
task, any start.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import plotly.graph_objects as go

from aeris.analysis.rollouts import (
    AT_GOAL,
    DETOUR,
    GOAL_OUTCOMES,
    LATE,
    OPTIMAL,
    TRAPPED,
    PolicyFates,
    Rollout,
    policy_fates,
    rollout,
)
from aeris.analysis.task_scores import optimal_solution, start_scores
from aeris.dashboard.data import RunData
from aeris.dashboard.panels.base import STYLE, Choice, Option, Output, Toggle
from aeris.dashboard.panels.grid import (
    ARROWS,
    ENV,
    add_goal,
    add_playback,
    cell_label,
    grid_figure,
    parse_cell,
    run_goal,
)
from aeris.dashboard.panels.tasks import SuitePanel, _protocol_choice, _suite
from aeris.domain.reports import SuiteTaskResult
from aeris.environments.four_rooms import Position
from aeris.environments.tabular import TabularDynamics

INK, MUTED = "#0b0b0b", "#52514e"
AGENT, OTHER_PROTOCOL = "#eb6834", "#6250d6"  # categorical slots 2 and 7
REWARD_SCALE = [[0.0, "#fcfcfb"], [1.0, "#86b6ef"]]  # recessive: the paths carry the story
COUNT_SCALE = [[0.0, "#cde2fb"], [1.0, "#104281"]]  # sequential blue
SCORE_SCALE = [[0.0, "#b02b2b"], [0.5, "#f0efec"], [1.0, "#1c5cab"]]  # worse than random < 0 < 1
STATUS = {
    AT_GOAL: "#f0efec",
    OPTIMAL: "#0ca30c",
    DETOUR: "#fab219",
    LATE: "#ec835a",
    TRAPPED: "#d03b3b",
}
CYCLE_COLOURS = ("#2a78d6", "#eb6834", "#1baf7a")  # a map compares all pairs: three hues at most
OTHER_CYCLES = "#a3a29c"
COLOURINGS = ("outcome", "start score", "visits")


@dataclass(frozen=True)
class _Behaviour:
    """One suite task's saved policies, per evidence protocol, on the exact dynamics."""

    result: SuiteTaskResult
    protocol: str  # the protocol being shown
    dynamics: TabularDynamics
    policies: Mapping[str, np.ndarray]  # protocol -> [state] actions, in dynamics order
    rewards: np.ndarray  # [state]
    goal: int | None  # a goal task's goal: rollouts end there
    goal_cell: Position | None  # drawn for goal and distance tasks
    gamma: float
    max_steps: int

    @property
    def actions(self) -> np.ndarray:
        return self.policies[self.protocol]


def _behaviour(run: RunData, task_id: str, protocol: str) -> _Behaviour:
    report = _suite(run)
    assert report is not None
    result = report.result(task_id)
    dynamics = ENV.tabular_dynamics()
    order = [dynamics.index_of(state) for state in report.states]
    policies = {}
    for name, outcome in result.outcomes.items():
        actions = np.empty(dynamics.num_states, dtype=np.int64)
        actions[order] = outcome.actions
        policies[name] = actions
    rewards = np.empty(dynamics.num_states)
    rewards[[dynamics.index_of(p) for p in ENV.positions]] = ENV.state_rewards(result.task)
    goal_cell = ENV.goal_from_task(result.task) if result.family in ("goal", "distance") else None
    return _Behaviour(
        result=result,
        protocol=protocol,
        dynamics=dynamics,
        policies=policies,
        rewards=rewards,
        goal=dynamics.index_of(goal_cell) if result.family == "goal" and goal_cell else None,
        goal_cell=goal_cell,
        gamma=report.gamma,
        max_steps=report.max_steps,
    )


def _task_choice(runs: Sequence[RunData]) -> Choice:
    """Every suite task; the run's own training goal first if it has one."""
    report = _suite(runs[0])
    ids = tuple(t.task.task_id for t in report.tasks) if report else ()
    goal = run_goal(runs[0])
    preferred = ENV.goal_task(goal).task_id if goal is not None else None
    return Choice("task", "Task", ids, preferred if preferred in ids else (ids[0] if ids else None))


def _xy(dynamics: TabularDynamics, states: Sequence[int]) -> tuple[list[int], list[int]]:
    cells = [dynamics.states[s] for s in states]
    return [c[1] for c in cells], [c[0] for c in cells]


def _describe(path: Rollout, dynamics: TabularDynamics, shortest: int | None, limit: int) -> str:
    if path.goal_step is not None:
        text = f"reaches the goal in {path.goal_step} steps (shortest {shortest})"
        return text + (f", after the {limit}-step limit" if path.goal_step > limit else "")
    return f"{_loop(path.cycle, dynamics)} from step {path.cycle_start}"


def _loop(cycle: Sequence[int], dynamics: TabularDynamics) -> str:
    """A loop in words; a one-cell loop means every step bumps into a wall."""
    if len(cycle) == 1:
        return f"stays at {cell_label(dynamics.states[cycle[0]])}"
    return f"loops through {len(cycle)} cells"


@dataclass(frozen=True)
class RolloutReplay(SuitePanel):
    name = "rollout"
    title = "Rollout"
    scope = "run"
    description = (
        "The exact path of the zero-shot policy from any start, for any suite task, "
        "next to an optimal path; × marks a suboptimal action."
    )

    def options(self, runs: Sequence[RunData]) -> tuple[Option, ...]:
        starts = tuple(cell_label(p) for p in ENV.positions)
        return (
            _task_choice(runs),
            _protocol_choice(runs),
            Choice("start", "Start cell", starts, "1,1" if "1,1" in starts else starts[0]),
            Toggle("optimal", "Show an optimal path", True),
            Toggle("compare", "Overlay the other evidence protocol", False),
        )

    def render(self, runs: Sequence[RunData], options: Mapping[str, Any]) -> Output:
        b = _behaviour(runs[0], options["task"], options["protocol"])
        dynamics = b.dynamics
        start = dynamics.index_of(parse_cell(options["start"]))
        path = rollout(dynamics, b.actions, start, goal=b.goal)
        solution = optimal_solution(dynamics, b.rewards, gamma=b.gamma)
        cells = {dynamics.states[s]: float(r) for s, r in enumerate(b.rewards)}
        low, high = float(b.rewards.min()), float(b.rewards.max())
        fig = grid_figure(
            cells,
            zmin=low if low < high else low - 1,
            zmax=high,
            colorscale=REWARD_SCALE,
            colorbar_title="reward",
        )
        if options["optimal"]:
            best = np.argmax(solution.optimal_actions, axis=1)
            x, y = _xy(dynamics, rollout(dynamics, best, start, goal=b.goal).states)
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines",
                    line={"color": MUTED, "width": 2, "dash": "dash"},
                    name="an optimal path",
                    hoverinfo="skip",
                )
            )
        others = [p for p in b.result.outcomes if p != b.protocol]
        other: Rollout | None = None
        if options["compare"] and others:
            other = rollout(dynamics, b.policies[others[0]], start, goal=b.goal)
            x, y = _xy(dynamics, other.states)
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines",
                    line={"color": OTHER_PROTOCOL, "width": 2, "dash": "dot"},
                    name=others[0],
                    hoverinfo="skip",
                )
            )
        if b.goal_cell is not None:
            add_goal(fig, b.goal_cell)
        fig.add_trace(
            go.Scatter(
                x=[dynamics.states[start][1]],
                y=[dynamics.states[start][0]],
                mode="markers",
                marker={"symbol": "circle-open", "size": 18, "color": INK, "line": {"width": 2}},
                name="start",
                hoverinfo="skip",
            )
        )

        x, y = _xy(dynamics, path.states)
        hover = [
            f"step {i} · cell {cell_label(dynamics.states[s])}"
            + (
                f" · {ARROWS[path.actions[i]]}"
                + ("" if solution.optimal_actions[s, path.actions[i]] else " (suboptimal)")
                if i < path.steps
                else ""
            )
            for i, s in enumerate(path.states)
        ]
        wrong = [
            i
            for i in range(path.steps)
            if not solution.optimal_actions[path.states[i], path.actions[i]]
        ]
        first = len(fig.data)
        fig.add_trace(
            go.Scatter(
                x=x[:1],
                y=y[:1],
                mode="lines+markers",
                line={"color": AGENT, "width": 3},
                marker={"size": 6},
                name="agent",
                text=hover[:1],
                hovertemplate="%{text}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x[:1],
                y=y[:1],
                mode="markers",
                marker={"size": 16, "color": AGENT, "line": {"width": 2, "color": "white"}},
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[],
                y=[],
                mode="markers",
                marker={
                    "symbol": "x-thin",
                    "size": 12,
                    "color": STATUS[TRAPPED],
                    "line": {"width": 3},
                },
                name="suboptimal action",
                hoverinfo="skip",
            )
        )
        frames = []
        for t in range(len(path.states)):
            shown = [i for i in wrong if i < t]
            frames.append(
                go.Frame(
                    data=[
                        go.Scatter(x=x[: t + 1], y=y[: t + 1], text=hover[: t + 1]),
                        go.Scatter(x=[x[t]], y=[y[t]]),
                        go.Scatter(x=[x[i] for i in shown], y=[y[i] for i in shown]),
                    ],
                    traces=[first, first + 1, first + 2],
                    name=str(t),
                )
            )
        add_playback(fig, frames)

        shortest = (
            int(dynamics.shortest_path_lengths(b.goal)[start]) if b.goal is not None else None
        )
        score = start_scores(dynamics, b.actions, b.rewards, gamma=b.gamma)[start]
        parts = [
            _describe(path, dynamics, shortest, b.max_steps),
            f"{len(wrong)} suboptimal action{'s' * (len(wrong) != 1)}",
            f"start score {round(score, 2) + 0.0:.2f}" if np.isfinite(score) else "start score n/a",
        ]
        if other is not None:  # it may hide exactly under the agent's path
            same = other.states == path.states
            parts.append(f"{others[0]}: {'same path' if same else 'different path'}")
        fig.update_traces(showscale=False, selector={"type": "heatmap"})  # reward: on hover
        title = f"{b.result.task.task_id} · {b.protocol} · from {options['start']}"
        return STYLE.apply(fig, f"{title}<br><sup>{' · '.join(parts)}</sup>")


def _discrete_scale(colours: Sequence[str]) -> list[list[Any]]:
    """A heatmap colour scale with one flat band per integer code 0..n-1 (zmin=-0.5)."""
    n = len(colours)
    scale: list[list[Any]] = []
    for i, colour in enumerate(colours):
        scale += [[i / n, colour], [(i + 1) / n, colour]]
    return scale


def _cell_heatmap(
    fig: go.Figure,
    dynamics: TabularDynamics,
    values: np.ndarray,
    text: Sequence[str],
    **heatmap: Any,
) -> None:
    z = np.full((ENV.HEIGHT, ENV.WIDTH), np.nan)
    hover = np.full((ENV.HEIGHT, ENV.WIDTH), "", dtype=object)
    for index, (row, col) in enumerate(dynamics.states):
        z[row, col], hover[row, col] = values[index], text[index]
    fig.add_trace(go.Heatmap(z=z, text=hover, hovertemplate="%{text}<extra></extra>", **heatmap))


def _loop_reward(b: _Behaviour, cycle: Sequence[int]) -> float:
    """The mean reward per step the policy collects forever once it is in ``cycle``."""
    return float(b.rewards[list(cycle)].mean())


def _fate(fates: PolicyFates, dynamics: TabularDynamics, state: int, cell: str) -> str:
    """Hover text: where the policy takes one start state."""
    steps = int(fates.goal_steps[state])
    if steps >= 0:
        shortest = fates.shortest_steps
        return f"{cell}: goal in {steps} steps" + (
            f" (shortest {shortest[state]})" if shortest is not None else ""
        )
    loop = fates.cycles[fates.cycle_of[state]]
    return f"{cell}: {_loop(loop, dynamics)} from step {fates.transient[state]}"


@dataclass(frozen=True)
class FateMap(SuitePanel):
    name = "fate_map"
    title = "Fate map"
    scope = "run"
    description = (
        "Every start cell coloured by where the zero-shot policy takes it: to the goal "
        "(optimally, with a detour, too late) or into a loop; or by start score or visits."
    )

    def options(self, runs: Sequence[RunData]) -> tuple[Option, ...]:
        return (
            _task_choice(runs),
            _protocol_choice(runs),
            Choice("colour", "Colour by", COLOURINGS, "outcome"),
        )

    def render(self, runs: Sequence[RunData], options: Mapping[str, Any]) -> Output:
        b = _behaviour(runs[0], options["task"], options["protocol"])
        dynamics = b.dynamics
        fates = policy_fates(dynamics, b.actions, goal=b.goal)
        cells = [cell_label(s) for s in dynamics.states]
        fate_text = [_fate(fates, dynamics, s, cell) for s, cell in enumerate(cells)]
        fig = grid_figure()
        colour = options["colour"]
        if colour == "start score":
            scores = start_scores(dynamics, b.actions, b.rewards, gamma=b.gamma)
            text = [f"{t} · start score {v:.2f}" for t, v in zip(fate_text, scores, strict=True)]
            _cell_heatmap(
                fig,
                dynamics,
                np.clip(scores, -1, 1),
                text,
                zmin=-1,
                zmax=1,
                colorscale=SCORE_SCALE,
                colorbar={"title": "start score", "thickness": 12},
            )
            summary = f"mean start score {np.nanmean(scores):.2f}"
        elif colour == "visits":
            text = [
                f"{t} · entered from {v} other starts"
                for t, v in zip(fate_text, fates.visits, strict=True)
            ]
            _cell_heatmap(
                fig,
                dynamics,
                fates.visits.astype(float),
                text,
                zmin=0,
                zmax=max(1, int(fates.visits.max())),
                colorscale=COUNT_SCALE,
                colorbar={"title": "starts", "thickness": 12},
            )
            busiest = int(np.argmax(fates.visits))
            summary = (
                f"busiest cell {cells[busiest]}: entered from {fates.visits[busiest]} other starts"
            )
        elif b.goal is not None:
            outcomes = fates.goal_outcomes(max_steps=b.max_steps)
            codes = np.array([GOAL_OUTCOMES.index(o) for o in outcomes], dtype=float)
            text = [f"{t} · {o}" for t, o in zip(fate_text, outcomes, strict=True)]
            _cell_heatmap(
                fig,
                dynamics,
                codes,
                text,
                zmin=-0.5,
                zmax=len(GOAL_OUTCOMES) - 0.5,
                colorscale=_discrete_scale([STATUS[o] for o in GOAL_OUTCOMES]),
                colorbar={
                    "thickness": 12,
                    "tickvals": list(range(len(GOAL_OUTCOMES))),
                    "ticktext": list(GOAL_OUTCOMES),
                },
            )
            others = outcomes[np.arange(len(outcomes)) != b.goal]
            summary = " · ".join(
                f"{o} {np.mean(others == o):.0%}" for o in (OPTIMAL, DETOUR, LATE, TRAPPED)
            )
        else:
            summary = self._cycle_outcomes(fig, b, fates, fate_text)
        self._draw_cycles(fig, b, fates)
        if b.goal_cell is not None:
            add_goal(fig, b.goal_cell)
        title = f"{self.title} · {b.result.task.task_id} · {b.protocol}"
        fig = STYLE.apply(fig, f"{title}<br><sup>{summary}</sup>")
        fig.update_layout(
            legend={"orientation": "h", "x": 0, "y": -0.06}
        )  # clear of the colour bar
        return fig

    @staticmethod
    def _ranked_cycles(fates: PolicyFates) -> list[int]:
        """Cycles by the number of starts that end in them, most first."""
        counts = np.bincount(fates.cycle_of[fates.cycle_of >= 0], minlength=len(fates.cycles))
        return sorted(range(len(fates.cycles)), key=lambda c: (-counts[c], c))

    def _cycle_outcomes(
        self, fig: go.Figure, b: _Behaviour, fates: PolicyFates, fate_text: Sequence[str]
    ) -> str:
        """Colour starts by the loop they end in (the three largest; the rest as 'other')."""
        ranked = self._ranked_cycles(fates)
        shown = ranked[: len(CYCLE_COLOURS)]
        code = {cycle: rank for rank, cycle in enumerate(shown)}
        other = len(shown)
        codes = np.array([code.get(int(c), other) for c in fates.cycle_of], dtype=float)
        labels = [
            f"loop {rank + 1}: {_loop(fates.cycles[c], b.dynamics)}, "
            f"mean reward {_loop_reward(b, fates.cycles[c]):.2f}"
            for rank, c in enumerate(shown)
        ]
        colours = list(CYCLE_COLOURS[: len(shown)])
        if len(ranked) > len(shown):
            labels.append(f"{len(ranked) - len(shown)} other loops")
            colours.append(OTHER_CYCLES)
        text = [f"{t} · {labels[int(c)]}" for t, c in zip(fate_text, codes, strict=True)]
        _cell_heatmap(
            fig,
            b.dynamics,
            codes,
            text,
            zmin=-0.5,
            zmax=len(colours) - 0.5,
            colorscale=_discrete_scale(colours),
            colorbar={
                "thickness": 12,
                "tickvals": list(range(len(colours))),
                "ticktext": [
                    f"{label.split(':')[0]} · r̄ {_loop_reward(b, fates.cycles[c]):.2f}"
                    for label, c in zip(labels, shown, strict=False)
                ]
                + labels[len(shown) :],
            },
        )
        solution = optimal_solution(b.dynamics, b.rewards, gamma=b.gamma)
        best = policy_fates(b.dynamics, np.argmax(solution.optimal_actions, axis=1))
        agent = max(_loop_reward(b, c) for c in fates.cycles)
        optimal = max(_loop_reward(b, c) for c in best.cycles)
        loops = len(fates.cycles)
        return (
            f"{loops} loop{'s' * (loops != 1)} · best loop's mean reward {agent:.2f}"
            f" (an optimal policy's: {optimal:.2f})"
        )

    def _draw_cycles(self, fig: go.Figure, b: _Behaviour, fates: PolicyFates) -> None:
        """Each loop as a closed ink outline over its cells."""
        for rank, c in enumerate(self._ranked_cycles(fates)):
            cycle = list(fates.cycles[c])
            x, y = _xy(b.dynamics, cycle + cycle[:1])
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines+markers",
                    line={"color": INK, "width": 2},
                    marker={"size": 8, "color": INK, "line": {"width": 1.5, "color": "white"}},
                    name="loop",
                    legendgroup="loop",
                    showlegend=rank == 0,
                    hoverinfo="skip",
                )
            )
