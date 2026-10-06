"""PCA of each representation: how many directions carry its variance, and what the leading ones encode spatially."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from predictive_representations_rl.analysis.base import Analyzer
from predictive_representations_rl.analysis import plotting

NUM_SAVED_COMPONENTS = 10
NUM_MAPPED_COMPONENTS = 3
MAX_PLOTTED_COMPONENTS = 64


def pca(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Scores [N, k] and explained-variance ratios [k] of centred values, via SVD."""
    centred = values - values.mean(axis=0)
    u, singular, _ = np.linalg.svd(centred, full_matrices=False)
    variance = singular**2
    total = variance.sum()
    ratio = variance / total if total > 0 else np.zeros_like(variance)
    return u * singular, ratio


def participation_ratio(ratio: np.ndarray) -> float:
    """Effective dimensionality (sum λ)^2 / sum λ^2: 1 if one direction dominates, D if variance is spread evenly."""
    return float(ratio.sum() ** 2 / np.sum(ratio**2)) if ratio.any() else 0.0


class PCAAnalyzer(Analyzer):
    name = "pca"

    def run(self, representations: dict[str, np.ndarray], metadata: dict[str, np.ndarray], output_dir: Path) -> dict[str, Any]:
        output_dir.mkdir(parents=True, exist_ok=True)
        plotting.style()

        summary: dict[str, Any] = {}
        ratios: dict[str, np.ndarray] = {}
        for name, values in representations.items():
            scores, ratio = pca(values)
            cumulative = np.cumsum(ratio)
            ratios[name] = ratio
            np.savez(output_dir / f"{name}.npz", scores=scores[:, :NUM_SAVED_COMPONENTS], explained_variance_ratio=ratio)
            summary[name] = {
                "dims": int(values.shape[1]),
                "components_for_90pct": int(np.searchsorted(cumulative, 0.90) + 1),
                "components_for_95pct": int(np.searchsorted(cumulative, 0.95) + 1),
                "participation_ratio": round(participation_ratio(ratio), 2),
                "explained_variance_ratio_top10": [round(float(r), 4) for r in ratio[:NUM_SAVED_COMPONENTS]],
            }
            if "x" in metadata and "y" in metadata:
                self._spatial_map(name, scores, ratio, metadata["x"], metadata["y"], output_dir / f"map_{name}.png")

        self._explained_variance(ratios, output_dir / "explained_variance.png")
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        return summary

    def _explained_variance(self, ratios: dict[str, np.ndarray], path: Path) -> None:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6.4, 4))
        for color, (name, ratio) in zip(plotting.SERIES, ratios.items()):
            cumulative = np.cumsum(ratio)[:MAX_PLOTTED_COMPONENTS]
            components = np.arange(1, len(cumulative) + 1)
            ax.plot(components, cumulative, color=color, label=name)
        ax.axhline(0.9, color=plotting.AXIS, linewidth=1, linestyle=(0, (3, 3)))
        ax.annotate("90%", (1, 0.9), xytext=(2, 3), textcoords="offset points", fontsize=7, color=plotting.INK_MUTED)
        ax.set_xscale("log")
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("number of principal components")
        ax.set_ylabel("cumulative explained variance")
        ax.set_title("How many directions carry each representation's variance")
        ax.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)

    def _spatial_map(self, name: str, scores: np.ndarray, ratio: np.ndarray, x: np.ndarray, y: np.ndarray, path: Path) -> None:
        import matplotlib.pyplot as plt

        count = min(NUM_MAPPED_COMPONENTS, scores.shape[1])
        fig, axes = plt.subplots(1, count, figsize=(3.4 * count, 3.4), squeeze=False)
        for index, ax in enumerate(axes[0]):
            values = scores[:, index]
            limit = np.percentile(np.abs(values), 98) or 1.0
            points = ax.scatter(x, y, c=values, cmap=plotting.DIVERGING, vmin=-limit, vmax=limit, s=8, linewidths=0)
            ax.set_title(f"PC{index + 1} ({ratio[index]:.0%} of variance)")
            ax.set_xlabel("x")
            ax.set_ylabel("y" if index == 0 else "")
            ax.set_aspect("equal")
            ax.grid(False)
            fig.colorbar(points, ax=ax, shrink=0.8, label="PC score")
        fig.suptitle(f"{name}: leading principal components over the maze", color=plotting.INK, fontsize=11)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
