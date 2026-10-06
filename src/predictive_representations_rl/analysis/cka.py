"""Linear CKA between representations of the same probe states (Kornblith et al., 2019).

CKA compares how two representations arrange the probes, not their coordinates: it is invariant to rotations and
isotropic scaling of either representation, so representations from unrelated networks (FB vs Dreamer) and of
different widths can be compared. Linear CKA is biased upwards for wide representations relative to the number of
probes; read values against the `_untrained` controls and the observation baseline in the same matrix.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from predictive_representations_rl.analysis import plotting


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    """CKA with linear kernels, computed in feature space: ||Yc^T Xc||_F^2 / (||Xc^T Xc||_F ||Yc^T Yc||_F)."""
    x = x - x.mean(axis=0)
    y = y - y.mean(axis=0)
    cross = np.linalg.norm(y.T @ x) ** 2
    norm = np.linalg.norm(x.T @ x) * np.linalg.norm(y.T @ y)
    return float(cross / norm) if norm > 0 else 0.0


def cka_matrix(representations: dict[str, np.ndarray]) -> tuple[list[str], np.ndarray]:
    names = list(representations)
    matrix = np.eye(len(names))
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            matrix[i, j] = matrix[j, i] = linear_cka(representations[names[i]], representations[names[j]])
    return names, matrix


def write_matrix(names: list[str], matrix: np.ndarray, path: Path) -> None:
    with open(path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["representation", *names])
        for name, row in zip(names, matrix):
            writer.writerow([name, *(f"{value:.4f}" for value in row)])


def plot_matrix(names: list[str], matrix: np.ndarray, path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    plotting.style()
    size = 0.48 * len(names) + 3.2
    fig, ax = plt.subplots(figsize=(size + 1.2, size))
    image = ax.imshow(matrix, cmap=plotting.SEQUENTIAL, vmin=0, vmax=1)
    if len(names) <= 24:
        for i in range(len(names)):
            for j in range(len(names)):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                        color="#ffffff" if matrix[i, j] > 0.55 else plotting.INK)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=60, ha="right")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.grid(False)
    ax.set_title(title, fontsize=10)
    fig.colorbar(image, ax=ax, shrink=0.7, label="linear CKA")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
