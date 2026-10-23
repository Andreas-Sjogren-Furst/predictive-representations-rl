"""Optional, reproducible plots from saved numerical FB diagnostics."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def render_fb_action_alignment(report: dict, output_file: str | Path) -> Path:
    """Map whether FB's greedy choice improves its own target-Boltzmann policy.

    This is NOT a map of globally optimal actions for terminal navigation.
    The report is sufficient; no neural model or training environment is loaded.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from aeris.environments import FourRooms

    env = FourRooms()
    positions = report['state_positions']
    agreement = report['state_action_improvement_agreement']
    if len(positions) != len(env.positions) or len(agreement) != len(positions):
        raise ValueError('malformed statewise FB diagnostic')
    mat = np.full((env.HEIGHT, env.WIDTH), np.nan, dtype=np.float64)
    expected = set(env.positions)
    visited = set()
    for pos, good in zip(positions, agreement):
        point = (int(pos[0]), int(pos[1]))
        if point not in expected or point in visited or type(good) is not bool:
            raise ValueError('invalid or duplicate statewise FB diagnostic')
        visited.add(point)
        mat[point] = float(good)
    if visited != expected:
        raise ValueError('statewise FB diagnostic does not cover the maze')
    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    ax.set_facecolor('0.35')
    cmap = ListedColormap(['#dc7777', '#6eb79a'])
    ax.imshow(np.ma.masked_invalid(mat), cmap=cmap, vmin=0, vmax=1,
              interpolation='nearest', origin='upper')
    for r, c in expected:
        if [r, c] != report['goal']:
            ax.text(c, r, '✓' if bool(mat[r, c]) else '×', ha='center', va='center', fontsize=10)
    ax.plot(report['goal'][1], report['goal'][0], marker='*', markersize=16, color='#e4a72e')
    ax.set_xticks(range(env.WIDTH))
    ax.set_yticks(range(env.HEIGHT))
    ax.grid(color='0.85', lw=0.5)
    ax.set_xlim(-0.5, env.WIDTH - 0.5)
    ax.set_ylim(env.HEIGHT - 0.5, -0.5)
    ax.set_aspect('equal')
    ax.set_title('FB action agreement with exact target-policy improvement')
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    target = Path(output_file)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=145, bbox_inches='tight')
    plt.close(fig)
    return target
