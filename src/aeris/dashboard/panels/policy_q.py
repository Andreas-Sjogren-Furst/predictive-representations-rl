"""Q-values of a fixed policy: learned on the offline data, against exact and dataset-only.

Panels read ``analysis/policy_evaluation_v1.npz`` (``aeris analysis policy-evaluation
RUN``). Any suite task can be the reward: a learned F^pi is reward-free, so its
Q for reward r is F^pi(s, a) · z_r, with z_r read from the dataset's states.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from aeris.analysis.policy_evaluation import (
    PolicyEvaluation,
    dataset_q,
    effective_horizon,
    exact_q,
)
from aeris.analysis.state_labels import ACTION_NAMES
from aeris.dashboard.data import RunData
from aeris.dashboard.panels.base import (
    STYLE,
    Choice,
    Message,
    Option,
    Output,
    Panel,
    series_colour,
)
from aeris.dashboard.panels.grid import (
    COUNT_SCALE,
    ENV,
    INK,
    MUTED,
    add_grid,
    lock_aspect,
    run_goal,
)
from aeris.environments.four_rooms_tasks import task_suite

SCALES = ("shared", "each map its own")
_TASKS = {task.task_id: task for task in task_suite("four_rooms_v1").tasks}
_DYNAMICS = ENV.tabular_dynamics()


def _evaluation(run: RunData) -> PolicyEvaluation:
    assert run.policy_evaluation is not None
    return run.policy_evaluation


def _choices(runs: Sequence[RunData]) -> tuple[Choice, Choice]:
    evaluation = _evaluation(runs[0])
    labels = tuple(p.label for p in evaluation.policies)
    default = next((p.label for p in evaluation.policies if p.kind == "optimal"), labels[0])
    goal = run_goal(runs[0])
    task = ENV.goal_task(goal).task_id if goal is not None else next(iter(_TASKS))
    return (
        Choice("policy", "Fixed policy", labels, default),
        Choice(
            "task", "Reward (task)", tuple(_TASKS), task if task in _TASKS else next(iter(_TASKS))
        ),
    )


@dataclass(frozen=True)
class _Values:
    table: np.ndarray  # [S, A] the policy
    exact: np.ndarray  # [S, A] Q from the true dynamics
    tabular: np.ndarray  # [S, A] Q from the dataset's transitions only; NaN where unknown
    learned: np.ndarray  # [S, A] F^pi(s, a) · z_r
    covered: np.ndarray  # [S, A] bool

    @staticmethod
    def value(table: np.ndarray, q: np.ndarray) -> np.ndarray:
        """V(s) = sum_a pi(a|s) Q(s, a); NaN when a used action's Q is unknown."""
        used = table > 0
        unknown = np.any(used & np.isnan(q), axis=1)
        v = (table * np.nan_to_num(q)).sum(axis=1)
        v[unknown] = np.nan
        return v


def _values(run: RunData, options: Mapping[str, Any]) -> _Values:
    evaluation = _evaluation(run)
    index = evaluation.policy_index(options["policy"])
    rewards = ENV.state_rewards(_TASKS[options["task"]])
    table = evaluation.tables[index]
    gamma = evaluation.gamma
    return _Values(
        table=table,
        exact=exact_q(_DYNAMICS, table, rewards, gamma=gamma),
        tabular=dataset_q(_DYNAMICS, table, rewards, evaluation.covered, gamma=gamma),
        learned=evaluation.learned_q(index, rewards),
        covered=evaluation.covered,
    )


def _scale(learned: np.ndarray, exact: np.ndarray) -> float:
    """The least-squares slope through the origin: learned ≈ scale × exact."""
    denominator = float((exact * exact).sum())
    return float((learned * exact).sum() / denominator) if denominator > 0 else float("nan")


def _correlation(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


@dataclass(frozen=True)
class PolicyPanel(Panel):
    """Shared requirement: a saved policy-evaluation artifact."""

    def missing(self, runs: Sequence[RunData]) -> str | None:
        if runs[0].policy_evaluation is None:
            return f"No policy evaluation. Run: aeris analysis policy-evaluation {runs[0].root}"
        return None


@dataclass(frozen=True)
class PolicyQMaps(PolicyPanel):
    name = "policy_q_maps"
    title = "Value of a fixed policy"
    scope = "run"
    description = (
        "V^π(s) = Σ_a π(a|s) Q^π(s, a) of one fixed policy on one task's reward: exact from the "
        "true dynamics, from the dataset's transitions alone (blank where the data cannot tell), "
        "and learned on the dataset with the run's B frozen."
    )

    def options(self, runs: Sequence[RunData]) -> tuple[Option, ...]:
        return (*_choices(runs), Choice("scale", "Colour scale", SCALES, SCALES[0]))

    def render(self, runs: Sequence[RunData], options: Mapping[str, Any]) -> Output:
        values = _values(runs[0], options)
        maps = [
            ("exact", _Values.value(values.table, values.exact)),
            ("dataset only", _Values.value(values.table, values.tabular)),
            ("learned", _Values.value(values.table, values.learned)),
        ]
        shared = options["scale"] == SCALES[0]
        finite = np.concatenate([v[np.isfinite(v)] for _, v in maps])
        fig = make_subplots(
            rows=1, cols=3, subplot_titles=[name for name, _ in maps], horizontal_spacing=0.07
        )
        for col, (name, v) in enumerate(maps, start=1):
            low, high = (
                (float(finite.min()), float(finite.max()))
                if shared
                else (float(np.nanmin(v)), float(np.nanmax(v)))
                if np.isfinite(v).any()
                else (0.0, 1.0)
            )
            cells = {
                s: float(x) for s, x in zip(_DYNAMICS.states, v, strict=True) if np.isfinite(x)
            }
            add_grid(
                fig,
                cells,
                row=1,
                col=col,
                zmin=low,
                zmax=high if high > low else low + 1e-9,
                colorscale=COUNT_SCALE,
                colorbar={"thickness": 10, "x": 0.29 + 0.355 * (col - 1), "len": 0.8},
            )
            fig.data[-1].showscale = not shared or col == 3
            fig.data[
                -1
            ].hovertemplate = f"{name}, row %{{y}}, col %{{x}}: %{{z:.4g}}<extra></extra>"
        lock_aspect(fig, 3)
        exact, learned = maps[0][1], maps[2][1]
        known = int(np.isfinite(maps[1][1]).sum())
        horizon, error = effective_horizon(
            _DYNAMICS,
            values.table,
            ENV.state_rewards(_TASKS[options["task"]]),
            values.learned,
            gamma=_evaluation(runs[0]).gamma,
        )
        summary = (
            f"learned vs exact: correlation {_correlation(learned, exact):.2f}, "
            f"learned ≈ {_scale(learned, exact):.3g} × exact · dataset alone determines "
            f"{known}/{len(exact)} states<br>learned Q matches the exact Q of the first "
            f"{horizon} rewards best (mean error {error:.3g})"
        )
        fig = STYLE.apply(
            fig,
            f"{self.title} · {options['policy']} · reward {options['task']}"
            f"<br><sup>{summary}</sup>",
        )
        fig.update_layout(margin={**STYLE.margin, "t": 125})
        return fig


@dataclass(frozen=True)
class PolicyQScatter(PolicyPanel):
    name = "policy_q_scatter"
    title = "Learned against exact Q"
    scope = "run"
    description = (
        "Every (state, action) pair's learned Q^π against the exact one, coloured by action; "
        "hollow markers are pairs the dataset never contains."
    )

    def options(self, runs: Sequence[RunData]) -> tuple[Option, ...]:
        return _choices(runs)

    def render(self, runs: Sequence[RunData], options: Mapping[str, Any]) -> Output:
        values = _values(runs[0], options)
        exact, learned, covered = values.exact, values.learned, values.covered
        fig = go.Figure()
        labels = np.array([f"({r},{c})" for r, c in _DYNAMICS.states])
        for action, name in enumerate(ACTION_NAMES[: exact.shape[1]]):
            for seen in (True, False):
                mask = covered[:, action] == seen
                if not mask.any():
                    continue
                fig.add_trace(
                    go.Scatter(
                        x=exact[mask, action],
                        y=learned[mask, action],
                        mode="markers",
                        marker={
                            "color": series_colour(action),
                            "size": 7,
                            "symbol": "circle" if seen else "circle-open",
                            "line": {"width": 1.5, "color": series_colour(action)},
                        },
                        name=name if seen else f"{name} (not in data)",
                        legendgroup=name,
                        text=labels[mask],
                        hovertemplate=f"%{{text}} {name}: exact %{{x:.4g}}, learned %{{y:.4g}}"
                        "<extra></extra>",
                    )
                )
        slope = _scale(learned, exact)
        span = np.array([float(exact.min()), float(exact.max())])
        fig.add_trace(
            go.Scatter(
                x=span,
                y=slope * span,
                mode="lines",
                line={"color": MUTED, "dash": "dot"},
                name=f"learned = {slope:.3g} × exact",
                hoverinfo="skip",
            )
        )
        fig.update_xaxes(title="exact Q^π")
        fig.update_yaxes(title="learned Q^π")
        exact_best = exact >= exact.max(axis=1, keepdims=True) - 1e-9 * max(
            1.0, np.abs(exact).max()
        )
        agree = float(exact_best[np.arange(len(exact)), learned.argmax(axis=1)].mean())
        fig = STYLE.apply(
            fig,
            f"{self.title} · {options['policy']} · reward {options['task']}<br><sup>"
            f"correlation {_correlation(learned, exact):.2f} · learned best action is a best "
            f"action of exact Q^π in {agree:.0%} of states</sup>",
        )
        fig.update_layout(margin={**STYLE.margin, "t": 80})
        return fig


@dataclass(frozen=True)
class PolicyQTraining(PolicyPanel):
    name = "policy_q_training"
    title = "Policy-evaluation training"
    scope = "run"
    description = "The loss of every learned F^π along training, to judge whether it converged."

    def render(self, runs: Sequence[RunData], options: Mapping[str, Any]) -> Output:
        evaluation = _evaluation(runs[0])
        trained = [
            i for i in range(len(evaluation.policies)) if np.isfinite(evaluation.losses[i]).any()
        ]
        if not trained:
            return Message("No policy was trained (FB's own policy uses the run's F).")
        fig = go.Figure()
        for slot, index in enumerate(trained):
            loss = evaluation.losses[index]
            steps = np.linspace(evaluation.updates / len(loss), evaluation.updates, len(loss))
            fig.add_trace(
                go.Scatter(
                    x=steps,
                    y=loss,
                    mode="lines",
                    line={"color": series_colour(slot), "width": 2},
                    name=evaluation.policies[index].label,
                )
            )
        fig.update_xaxes(title="updates")
        fig.update_yaxes(title="loss (Bellman residual − diagonal)")
        fig = STYLE.apply(fig, f"{self.title} · B frozen, {evaluation.updates:,} updates each")
        fig.update_layout(legend={"font": {"color": INK}})
        return fig
