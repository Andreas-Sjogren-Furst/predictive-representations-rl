"""Files and figures for linear probes; the probing itself lives in analysis.probing."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from predictive_representations_rl.analysis.probing import probe, probe_table, skip_reason  # noqa: F401  (re-exported)

from predictive_representations_rl.analysis import plotting
from predictive_representations_rl.analysis.base import Analyzer

FIELDS = ["representation", "factor", "kind", "metric", "score", "score_std", "chance"]


class LinearProbeAnalyzer(Analyzer):
    name = "linear_probe"

    def run(self, representations: dict[str, np.ndarray], metadata: dict[str, np.ndarray], output_dir: Path) -> dict[str, Any]:
        output_dir.mkdir(parents=True, exist_ok=True)
        rows, skipped = probe_table(representations, metadata, metadata.get("episode"))

        write_rows(rows, output_dir / "linear_probe.csv")
        plot_scores(rows, output_dir / "linear_probe.png", "Linear probe: factor recovered from each representation")

        summary: dict[str, Any] = {rep: {} for rep in representations}
        for row in rows:
            summary[row["representation"]][row["factor"]] = round(row["score"], 4)
        summary["skipped_factors"] = skipped
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        return summary


def write_rows(rows: list[dict[str, Any]], path: Path) -> None:
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in FIELDS})


def read_rows(path: Path) -> list[dict[str, Any]]:
    with open(path) as file:
        return [{**row, "score": float(row["score"]), "score_std": float(row["score_std"]), "chance": float(row["chance"])}
                for row in csv.DictReader(file)]


def plot_scores(rows: list[dict[str, Any]], path: Path, title: str) -> None:
    """Heatmap representation x factor. Cells show the score; colour is clipped to [0, 1] (R^2 can be negative)."""
    import matplotlib.pyplot as plt

    plotting.style()
    representations = list(dict.fromkeys(row["representation"] for row in rows))
    factors = list(dict.fromkeys(row["factor"] for row in rows))
    kinds = {row["factor"]: row["kind"] for row in rows}
    lookup = {(row["representation"], row["factor"]): row["score"] for row in rows}
    grid = np.array([[lookup.get((rep, factor), np.nan) for factor in factors] for rep in representations])

    fig, ax = plt.subplots(figsize=(0.62 * len(factors) + 2.6, 0.42 * len(representations) + 1.9))
    image = ax.imshow(np.clip(grid, 0, 1), cmap=plotting.SEQUENTIAL, vmin=0, vmax=1, aspect="auto")
    for i in range(len(representations)):
        for j in range(len(factors)):
            if np.isfinite(grid[i, j]):
                dark = np.clip(grid[i, j], 0, 1) > 0.55
                text = f"{grid[i, j]:.2f}" if grid[i, j] >= 0 else "<0"  # exact values are in the CSV
                ax.text(j, i, text, ha="center", va="center", fontsize=7, color="#ffffff" if dark else plotting.INK)
    ax.set_xticks(range(len(factors)))
    ax.set_xticklabels([f"{f} (bal. acc)" if kinds[f] == "categorical" else f for f in factors], rotation=45, ha="right")
    ax.set_yticks(range(len(representations)))
    ax.set_yticklabels(representations)
    ax.grid(False)
    ax.set_title(f"{title}\nR² for continuous factors (<0: worse than the mean), balanced accuracy for (bal. acc)", fontsize=10)
    fig.colorbar(image, ax=ax, shrink=0.8, label="score")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
