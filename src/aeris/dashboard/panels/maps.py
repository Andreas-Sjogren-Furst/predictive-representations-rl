"""Four Rooms maps: the saved training-goal policy and a recorded trajectory."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import plotly.graph_objects as go

from aeris.analysis.goal_policy import build_goal_oracle
from aeris.analysis.rollouts import policy_fates
from aeris.dashboard.data import RunData
from aeris.dashboard.panels.base import (
    STYLE,
    Choice,
    Message,
    Number,
    Option,
    Output,
    Panel,
    Toggle,
)
from aeris.dashboard.panels.behaviour import add_actions, add_goal_outcomes, finish_map
from aeris.dashboard.panels.grid import (
    ENV,
    add_goal,
    add_playback,
    grid_figure,
    is_four_rooms,
    run_goal,
)


def _require_four_rooms(runs: Sequence[RunData]) -> str | None:
    if not is_four_rooms(runs[0]) or run_goal(runs[0]) is None:
        return "Map panels support single-goal Four Rooms runs."
    return None


@dataclass(frozen=True)
class PolicyMap(Panel):
    name = "policy_map"
    title = "Policy map"
    scope = "run"
    description = (
        "The training goal's saved policy (z = B(goal)): the action in every cell, suboptimal "
        "ones marked, over where it takes each start (optimal, detour, late, trapped)."
    )

    def options(self, runs: Sequence[RunData]) -> tuple[Option, ...]:
        labels = tuple(runs[0].snapshots)
        return (
            Choice("snapshot", "Snapshot", labels, "final" if "final" in labels else None),
            Toggle("fates", "Colour cells by outcome", True),
            Toggle("check_optimal", "Mark suboptimal actions", True),
        )

    def missing(self, runs: Sequence[RunData]) -> str | None:
        return _require_four_rooms(runs) or (
            None if runs[0].snapshots else "No policy snapshots saved."
        )

    def render(self, runs: Sequence[RunData], options: Mapping[str, Any]) -> Output:
        run, goal = runs[0], run_goal(runs[0])
        assert goal is not None
        dynamics = ENV.tabular_dynamics()
        actions = np.asarray(run.snapshots[options["snapshot"]].actions_for(dynamics.states))
        oracle = build_goal_oracle(dynamics, goal, gamma=0.99)
        optimal = oracle.optimal.optimal_actions
        others = np.arange(len(actions)) != oracle.goal_index
        chosen = optimal[np.arange(len(actions)), actions]
        parts = [f"optimal choices {chosen[others].mean():.0%}"]
        fig = grid_figure()
        if options["fates"]:
            fates = policy_fates(dynamics, actions, goal=oracle.goal_index)
            parts.append(add_goal_outcomes(fig, dynamics, fates, max_steps=_max_steps(run)))
        add_actions(
            fig,
            dynamics,
            actions,
            optimal if options["check_optimal"] else None,
            skip=oracle.goal_index,
        )
        add_goal(fig, goal)
        title = f"{self.title} · {ENV.goal_task(goal).task_id} · {options['snapshot']}"
        return finish_map(fig, title, " · ".join(parts))


def _max_steps(run: RunData) -> int:
    """The evaluation's step limit, which separates a detour from arriving late."""
    limit = run.setting("evaluation.max_steps")
    return limit if isinstance(limit, int) else ENV.max_episode_steps


@dataclass(frozen=True)
class TrajectoryAnimation(Panel):
    name = "trajectory"
    title = "Trajectory"
    scope = "run"
    description = "Replay a recorded evaluation episode step by step."

    def options(self, runs: Sequence[RunData]) -> tuple[Option, ...]:
        checkpoints = tuple(dict.fromkeys(t.checkpoint_id for t in runs[0].trajectories))
        return (
            Choice(
                "checkpoint", "Checkpoint", checkpoints, checkpoints[-1] if checkpoints else None
            ),
            Number("episode", "Episode (index among saved)", 0, minimum=0, maximum=1000),
        )

    def missing(self, runs: Sequence[RunData]) -> str | None:
        return _require_four_rooms(runs) or (
            None if runs[0].trajectories else "No trajectories saved."
        )

    def render(self, runs: Sequence[RunData], options: Mapping[str, Any]) -> Output:
        run, goal = runs[0], run_goal(runs[0])
        assert goal is not None
        episodes = [t for t in run.trajectories if t.checkpoint_id == options["checkpoint"]]
        if not episodes:
            return Message("No trajectory for this checkpoint.")
        trajectory = episodes[min(int(options["episode"]), len(episodes) - 1)]
        path = [ENV.position_of(trajectory.initial_observation)]
        path += [ENV.position_of(step.next_observation) for step in trajectory.steps]
        rows, cols = [p[0] for p in path], [p[1] for p in path]
        fig = grid_figure()
        add_goal(fig, goal)
        fig.add_trace(
            go.Scatter(x=cols[:1], y=rows[:1], mode="lines", line={"color": "#4c78a8"}, name="path")
        )
        fig.add_trace(
            go.Scatter(
                x=cols[:1],
                y=rows[:1],
                mode="markers",
                marker={"size": 16, "color": "#4c78a8"},
                name="agent",
            )
        )
        trail, agent = len(fig.data) - 2, len(fig.data) - 1
        frames = [
            go.Frame(
                data=[
                    go.Scatter(x=cols[: t + 1], y=rows[: t + 1]),
                    go.Scatter(x=[cols[t]], y=[rows[t]]),
                ],
                traces=[trail, agent],
                name=str(t),
            )
            for t in range(len(path))
        ]
        add_playback(fig, frames)
        outcome = "reached goal" if trajectory.success else "not reached"
        episode = f"episode {trajectory.episode_index}"
        title = f"{self.title} · {trajectory.checkpoint_id} · {episode} · {outcome}"
        return STYLE.apply(fig, title)
