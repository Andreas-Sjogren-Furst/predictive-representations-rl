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
    COUNT_SCALE,
    ENV,
    INK,
    MUTED,
    SCORE_SCALE,
    add_goal,
    add_playback,
    cell_label,
    grid_figure,
    parse_cell,
    reward_colours,
    run_goal,
)
from aeris.dashboard.panels.tasks import SuitePanel, _fmt, _protocol_choice, _suite
from aeris.domain.reports import SuiteTaskResult, ZeroShotSuiteReport
from aeris.environments.four_rooms import Position
from aeris.environments.tabular import TabularDynamics

AGENT, OTHER_PROTOCOL = "#eb6834", "#6250d6"  # categorical slots 2 and 7
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
LOWEST = "lowest-scoring task"  # a task choice resolved per run and evidence protocol


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
    result = report.result(_lowest(report, protocol) if task_id == LOWEST else task_id)
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


def _lowest(report: ZeroShotSuiteReport, protocol: str) -> str:
    """The task with the lowest normalized score under ``protocol``."""
    scored = [
        (score, t.task.task_id)
        for t in report.tasks
        if (score := t.outcomes[protocol].normalized_score) is not None
    ]
    return min(scored)[1] if scored else report.tasks[0].task.task_id


def _task_choice(runs: Sequence[RunData], prefer: str | None = None) -> Choice:
    """Every suite task, after the lowest-scoring one; by default the ``prefer`` task, else the
    run's own training goal, else the first task."""
    report = _suite(runs[0])
    ids = tuple(t.task.task_id for t in report.tasks) if report else ()
    goal = run_goal(runs[0])
    if prefer is None and goal is not None:
        prefer = ENV.goal_task(goal).task_id
    default = prefer if prefer in ids else (ids[0] if ids else None)
    return Choice("task", "Task", (LOWEST, *ids) if ids else (), default)


def _task_title(b: _Behaviour, options: Mapping[str, Any]) -> str:
    task = b.result.task.task_id
    return f"{task} (lowest score)" if options["task"] == LOWEST else task


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
        fig = grid_figure(cells, colorbar_title="reward", **reward_colours(b.rewards))
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
        title = f"{_task_title(b, options)} · {b.protocol} · from {options['start']}"
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
        "(optimally, with a detour, too late) or into a loop; or by start score or visits. "
        "Optionally shows the action in every cell."
    )

    def options(self, runs: Sequence[RunData]) -> tuple[Option, ...]:
        return (
            _task_choice(runs),
            _protocol_choice(runs),
            Choice("colour", "Colour by", COLOURINGS, "outcome"),
            Toggle("actions", "Show actions (instead of loops)", False),
        )

    def render(self, runs: Sequence[RunData], options: Mapping[str, Any]) -> Output:
        b = _behaviour(runs[0], options["task"], options["protocol"])
        dynamics = b.dynamics
        fates = policy_fates(dynamics, b.actions, goal=b.goal)
        cells = [cell_label(s) for s in dynamics.states]
        fate_text = fate_texts(fates, dynamics)
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
            summary = add_goal_outcomes(fig, dynamics, fates, max_steps=b.max_steps)
        else:
            summary = self._cycle_outcomes(fig, b, fates, fate_text)
        if options["actions"]:
            solution = optimal_solution(dynamics, b.rewards, gamma=b.gamma)
            add_actions(fig, dynamics, b.actions, solution.optimal_actions, skip=b.goal)
        else:
            add_loops(fig, dynamics, fates)
        if b.goal_cell is not None:
            add_goal(fig, b.goal_cell)
        title = f"{self.title} · {_task_title(b, options)} · {b.protocol}"
        return finish_map(fig, title, summary)

    def _cycle_outcomes(
        self, fig: go.Figure, b: _Behaviour, fates: PolicyFates, fate_text: Sequence[str]
    ) -> str:
        """Colour starts by the loop they end in (the three largest; the rest as 'other')."""
        ranked = _ranked_cycles(fates)
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
        return loop_summary(b, fates)


def loop_summary(b: _Behaviour, fates: PolicyFates) -> str:
    """How many loops the policy ends in, and how many starts end in an optimal one: a loop
    collecting as much reward per step as the best loop of an optimal policy."""
    solution = optimal_solution(b.dynamics, b.rewards, gamma=b.gamma)
    best = policy_fates(b.dynamics, np.argmax(solution.optimal_actions, axis=1))
    optimal = max(_loop_reward(b, c) for c in best.cycles)
    rewards = np.array([_loop_reward(b, c) for c in fates.cycles])
    reached = rewards[fates.cycle_of[fates.cycle_of >= 0]] >= optimal - 1e-9
    loops = len(fates.cycles)
    text = (
        f"{loops} loop{'s' * (loops != 1)} · {reached.mean():.0%} of starts reach an "
        f"optimal loop (r̄ {optimal:.2f})"
    )
    if rewards.max() < optimal - 1e-9:
        text += f", the agent's best: {rewards.max():.2f}"
    return text


def _ranked_cycles(fates: PolicyFates) -> list[int]:
    """Cycles by the number of starts that end in them, most first."""
    counts = np.bincount(fates.cycle_of[fates.cycle_of >= 0], minlength=len(fates.cycles))
    return sorted(range(len(fates.cycles)), key=lambda c: (-counts[c], c))


def fate_texts(fates: PolicyFates, dynamics: TabularDynamics) -> list[str]:
    """Hover text for every start state: where the policy takes it."""
    return [_fate(fates, dynamics, s, cell_label(cell)) for s, cell in enumerate(dynamics.states)]


def add_goal_outcomes(
    fig: go.Figure, dynamics: TabularDynamics, fates: PolicyFates, *, max_steps: int
) -> str:
    """Colour every start by its goal outcome; returns the share of each outcome."""
    outcomes = fates.goal_outcomes(max_steps=max_steps)
    codes = np.array([GOAL_OUTCOMES.index(o) for o in outcomes], dtype=float)
    text = [f"{t} · {o}" for t, o in zip(fate_texts(fates, dynamics), outcomes, strict=True)]
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
    others = outcomes[np.arange(len(outcomes)) != fates.goal]
    return " · ".join(f"{o} {np.mean(others == o):.0%}" for o in (OPTIMAL, DETOUR, LATE, TRAPPED))


def add_actions(
    fig: go.Figure,
    dynamics: TabularDynamics,
    actions: np.ndarray,
    optimal: np.ndarray | None,
    *,
    skip: int | None = None,
) -> None:
    """The action in every cell as an ink arrow; a suboptimal one white on an ink disc.

    ``optimal`` is the [state, action] mask of optimal actions, or None to mark nothing.
    """
    states = [s for s in range(dynamics.num_states) if s != skip]
    wrong = [s for s in states if optimal is not None and not optimal[s, actions[s]]]
    x, y = _xy(dynamics, wrong)
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            marker={"size": 19, "color": INK},
            name="suboptimal action",
            showlegend=bool(wrong),
            hoverinfo="skip",
        )
    )
    for group, colour in (([s for s in states if s not in wrong], INK), (wrong, "white")):
        x, y = _xy(dynamics, group)
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="text",
                text=[ARROWS[int(actions[s])] for s in group],
                textfont={"size": 15, "color": colour},
                showlegend=False,
                hoverinfo="skip",
            )
        )


def finish_map(fig: go.Figure, title: str, summary: str) -> go.Figure:
    """Title with a summary line, and the legend below the grid, clear of the colour bar."""
    fig = STYLE.apply(fig, f"{title}<br><sup>{summary}</sup>")
    fig.update_layout(legend={"orientation": "h", "x": 0, "y": -0.06})
    return fig


def add_loops(
    fig: go.Figure, dynamics: TabularDynamics, fates: PolicyFates, *, rings: bool = False
) -> None:
    """Each loop as a closed ink outline over its cells; or, with ``rings``, its cells circled,
    leaving the arrows inside them readable."""
    if rings:
        cells = sorted({s for cycle in fates.cycles for s in cycle})
        x, y = _xy(dynamics, cells)
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                marker={"symbol": "circle-open", "size": 26, "color": INK, "line": {"width": 2}},
                name="loop (where it ends up)",
                showlegend=bool(cells),
                hoverinfo="skip",
            )
        )
        return
    for rank, c in enumerate(_ranked_cycles(fates)):
        cycle = list(fates.cycles[c])
        x, y = _xy(dynamics, cycle + cycle[:1])
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


@dataclass(frozen=True)
class TaskPolicy(SuitePanel):
    name = "task_policy"
    title = "Task policy"
    scope = "run"
    description = (
        "One task's reward in every cell with the action the agent inferred for it: suboptimal "
        "actions white on ink, the loops it ends in circled."
    )

    def options(self, runs: Sequence[RunData]) -> tuple[Option, ...]:
        report = _suite(runs[0])
        rooms = [t.task.task_id for t in report.tasks if t.family == "room"] if report else []
        return (
            _task_choice(runs, prefer=rooms[0] if rooms else None),
            _protocol_choice(runs),
            Toggle("check_optimal", "Mark suboptimal actions", True),
        )

    def render(self, runs: Sequence[RunData], options: Mapping[str, Any]) -> Output:
        b = _behaviour(runs[0], options["task"], options["protocol"])
        dynamics = b.dynamics
        fates = policy_fates(dynamics, b.actions, goal=b.goal)
        text = [
            f"{t} · reward {r:.2f}"
            for t, r in zip(fate_texts(fates, dynamics), b.rewards, strict=True)
        ]
        fig = grid_figure()
        _cell_heatmap(
            fig,
            dynamics,
            b.rewards,
            text,
            colorbar={"title": "reward", "thickness": 12},
            **reward_colours(b.rewards),
        )
        optimal = optimal_solution(dynamics, b.rewards, gamma=b.gamma).optimal_actions
        add_loops(fig, dynamics, fates, rings=True)
        add_actions(
            fig, dynamics, b.actions, optimal if options["check_optimal"] else None, skip=b.goal
        )
        if b.goal_cell is not None:
            add_goal(fig, b.goal_cell)
        outcome = b.result.outcomes[b.protocol]
        states = np.arange(dynamics.num_states) != (-1 if b.goal is None else b.goal)
        wrong = ~optimal[np.arange(dynamics.num_states), b.actions]
        parts = [
            f"score {_fmt(outcome.normalized_score)}",
            f"{wrong[states].mean():.0%} suboptimal actions",
        ]
        if outcome.goal_success is not None:
            parts.append(f"goal reached from {_fmt(outcome.goal_success, percent=True)} of starts")
        elif fates.cycles:
            parts.append(loop_summary(b, fates))
        if outcome.rewarded_samples == 0:
            parts.append("no rewarded evidence")
        return finish_map(fig, f"{_task_title(b, options)} · {b.protocol}", " · ".join(parts))
