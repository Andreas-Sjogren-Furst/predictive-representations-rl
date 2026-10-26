"""Per-goal zero-shot success maps from a saved ZeroShotReport."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from aeris.domain.reports import ZeroShotReport
from aeris.environments import FourRooms
from aeris.visualization._plotting import WALL_COLOR, pyplot, save_figure, style_grid_axes


def render_zero_shot_success(report: ZeroShotReport, output_file: str | Path) -> Path:
    """One panel per evidence protocol: each goal cell coloured by its success rate.

    Cells that were not evaluated as goals stay blank; × marks a goal whose
    evidence contained no rewarded sample (the task was invisible to inference).
    """
    env = FourRooms()
    protocols = list(report.evidence_protocols)
    fig, axes = pyplot().subplots(
        1, len(protocols), figsize=(5.6 * len(protocols), 5.2), squeeze=False
    )
    image = None
    for ax, protocol in zip(axes[0], protocols, strict=True):
        grid = np.full((env.HEIGHT, env.WIDTH), np.nan)
        for task in report.tasks:
            outcome = task.outcomes[protocol]
            row, col = task.goal
            grid[row, col] = outcome.success_rate
            if outcome.rewarded_samples == 0:
                ax.text(col, row, "×", ha="center", va="center", fontsize=9)
        ax.set_facecolor(WALL_COLOR)
        image = ax.imshow(
            np.ma.masked_invalid(grid), cmap="viridis", vmin=0, vmax=1, interpolation="nearest"
        )
        style_grid_axes(ax, env)
        ax.set_title(f"{protocol} · mean success {report.mean_success_rate[protocol]:.1%}")
    fig.colorbar(image, ax=axes[0].tolist(), shrink=0.8, label="goal success rate")
    fig.suptitle(f"Zero-shot goal reaching · {report.run_id} · {report.episodes} episodes per goal")
    return save_figure(fig, output_file)
