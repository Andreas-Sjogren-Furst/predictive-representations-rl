"""Plot deterministic policy diagnostics from saved artifacts and exact models.

These are Four Rooms-specific renderings, not part of the generic RL runner.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from aeris.analysis.four_rooms import FourRoomsOracle, TabularPolicyAnalysis
from aeris.analysis.tabular import PolicyDiagnostics
from aeris.environments import FourRooms
from aeris.visualization._plotting import ACTION_ARROWS, draw_grid, pyplot, save_figure

OPTIMAL_COLOR = "#16803c"
SUBOPTIMAL_COLOR = "#d12626"


def render_optimal_policy(oracle: FourRoomsOracle, output_file: str | Path) -> Path:
    """Display *all* optimal choices per state; ties are not arbitrarily discarded."""
    fig, ax = pyplot().subplots(figsize=(7, 7))
    draw_grid(ax, FourRooms(), oracle.goal)
    for i, (row, col) in enumerate(oracle.positions):
        if i == oracle.goal_index:
            continue
        symbols = "".join(
            ACTION_ARROWS[a] for a in np.flatnonzero(oracle.optimal.optimal_actions[i])
        )
        ax.text(col, row, symbols, ha="center", va="center", fontsize=8.5)
    ax.set_title("Exact optimal actions · all equally optimal actions shown")
    return save_figure(fig, output_file, dpi=150)


def render_policy_gap(
    oracle: FourRoomsOracle,
    actions: np.ndarray,
    diagnostics: PolicyDiagnostics,
    output_file: str | Path,
    *,
    label: str = "Policy",
) -> Path:
    """Green = chosen action optimal (including ties), red = suboptimal."""
    from matplotlib.lines import Line2D

    fig, ax = pyplot().subplots(figsize=(7, 7))
    draw_grid(ax, FourRooms(), oracle.goal)
    for i, (row, col) in enumerate(oracle.positions):
        if i == oracle.goal_index:
            continue
        color = OPTIMAL_COLOR if diagnostics.optimal_by_state[i] else SUBOPTIMAL_COLOR
        ax.text(
            col,
            row,
            ACTION_ARROWS[int(actions[i])],
            ha="center",
            va="center",
            fontsize=14,
            fontweight="bold",
            color=color,
        )
    ax.set_title(
        f"{label} · optimal choices {diagnostics.optimal_action_fraction:.0%}"
        f" · all-start success {diagnostics.rollout_success_fraction:.0%}"
    )
    legend = [
        Line2D([], [], marker="$✓$", linestyle="None", color=OPTIMAL_COLOR, label="Optimal choice"),
        Line2D(
            [],
            [],
            marker="$×$",
            linestyle="None",
            color=SUBOPTIMAL_COLOR,
            label="Suboptimal choice",
        ),
    ]
    ax.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    return save_figure(fig, output_file, dpi=150)


def render_tabular_policy_figures(
    analysis: TabularPolicyAnalysis, output_dir: str | Path
) -> tuple[Path, ...]:
    """Render the optimal policy and one policy-gap map per analyzed snapshot."""
    output = Path(output_dir)
    made = [render_optimal_policy(analysis.oracle, output / "optimal_policy.png")]
    for phase, snapshot in analysis.snapshots.items():
        made.append(
            render_policy_gap(
                analysis.oracle,
                snapshot.actions,
                snapshot.diagnostics,
                output / f"policy_gap_{phase}.png",
                label=phase.capitalize(),
            )
        )
    return tuple(made)
