"""Files and figures for the linear CKA comparison; the computation lives in rl_lab.analysis.cka."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from rl_lab.analysis.cka import cka_matrix, linear_cka  # noqa: F401  (re-exported)

from predictive_representations_rl.analysis import plotting


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
