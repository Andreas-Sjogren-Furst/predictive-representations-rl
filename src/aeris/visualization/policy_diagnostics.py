"""Plot deterministic policy diagnostics from saved artifacts and exact models.

These are Four Rooms-specific renderings, not part of the generic RL runner.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np

from aeris.analysis.four_rooms import FourRoomsOracle
from aeris.analysis.tabular import PolicyDiagnostics
from aeris.environments import FourRooms
from aeris.visualization.four_rooms import _draw_map, _plot_lib

_ARROWS = {0: "↑", 1: "↓", 2: "←", 3: "→"}


def render_optimal_policy(oracle: FourRoomsOracle, output_file: str | Path) -> Path:
    """Display *all* optimal choices per state; ties are not arbitrarily discarded."""
    plt, _ = _plot_lib()
    env = FourRooms()
    fig, ax = plt.subplots(figsize=(7, 7))
    _draw_map(ax, env, oracle.goal)
    for i, (row, col) in enumerate(oracle.positions):
        if i == oracle.goal_index:
            continue
        symbols = "".join(_ARROWS[a] for a in np.flatnonzero(oracle.optimal.optimal_actions[i]))
        ax.text(col, row, symbols, ha="center", va="center", fontsize=8.5)
    ax.set_title("Exact optimal actions · all equally optimal actions shown")
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def render_policy_gap(
    oracle: FourRoomsOracle,
    actions: np.ndarray,
    diagnostics: PolicyDiagnostics,
    output_file: str | Path,
    *,
    label: str = "Policy",
) -> Path:
    """Green = chosen action optimal (including ties), red = suboptimal."""
    plt, _ = _plot_lib()
    from matplotlib.lines import Line2D
    env = FourRooms()
    fig, ax = plt.subplots(figsize=(7, 7))
    _draw_map(ax, env, oracle.goal)
    for i, (row, col) in enumerate(oracle.positions):
        if i == oracle.goal_index:
            continue
        ax.text(col, row, _ARROWS[int(actions[i])], ha="center", va="center",
                fontsize=14, fontweight="bold",
                color="#16803c" if diagnostics.optimal_by_state[i] else "#d12626")
    ax.set_title(f"{label} · optimal choices {diagnostics.optimal_action_fraction:.0%}"
                 f" · all-start success {diagnostics.rollout_success_fraction:.0%}")
    ax.legend(handles=[Line2D([], [], marker="$✓$", linestyle="None", color="#16803c", label="Optimal choice"),
                       Line2D([], [], marker="$×$", linestyle="None", color="#d12626", label="Suboptimal choice")],
              loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path
