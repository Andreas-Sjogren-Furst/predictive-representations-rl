"""Linear probes: how well a linear readout of each representation recovers each ground-truth factor.

Continuous factors: ridge regression, cross-validated R^2. Categorical factors (rooms, rewarded-or-not): class-balanced
logistic regression, balanced accuracy, so chance is 1/k however imbalanced the classes are. Folds are split by
episode when the probe set records it, because windows from the same episode overlap and would otherwise leak
between train and test; categorical folds are also stratified by class.
"""

from __future__ import annotations

import csv
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.model_selection import GroupKFold, KFold, StratifiedGroupKFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from predictive_representations_rl.analysis import plotting
from predictive_representations_rl.analysis.base import Analyzer, factor_columns

NUM_FOLDS = 5
# A class needs this many probes, from at least NUM_FOLDS episodes, to be probed; otherwise the factor is skipped.
MIN_CLASS_SIZE = 20
RIDGE_ALPHAS = np.logspace(-3, 3, 13)
FIELDS = ["representation", "factor", "kind", "metric", "score", "score_std", "chance"]


def skip_reason(target: np.ndarray, kind: str, groups: np.ndarray | None) -> str | None:
    """Why a factor cannot be probed reliably, or None."""
    if np.ptp(target) == 0:
        return "constant"
    if kind == "categorical":
        for label in np.unique(target):
            members = target == label
            episodes = len(np.unique(groups[members])) if groups is not None else NUM_FOLDS
            if members.sum() < MIN_CLASS_SIZE or episodes < NUM_FOLDS:
                return f"class {label} has {members.sum()} probes from {episodes} episodes"
    return None


def probe(values: np.ndarray, target: np.ndarray, kind: str, groups: np.ndarray | None) -> dict[str, Any]:
    grouped = groups is not None and len(np.unique(groups)) >= NUM_FOLDS

    if kind == "categorical":
        folds = StratifiedGroupKFold(NUM_FOLDS, shuffle=True, random_state=0) if grouped else StratifiedKFold(NUM_FOLDS, shuffle=True, random_state=0)
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
        metric = "balanced_accuracy"
        chance = 1.0 / len(np.unique(target))
    else:
        folds = GroupKFold(NUM_FOLDS) if grouped else KFold(NUM_FOLDS, shuffle=True, random_state=0)
        model = make_pipeline(StandardScaler(), RidgeCV(alphas=RIDGE_ALPHAS))
        metric = "r2"
        chance = 0.0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        scores = cross_val_score(model, values, target, cv=folds, groups=groups if grouped else None, scoring=metric)

    return {"metric": metric, "score": float(scores.mean()), "score_std": float(scores.std()), "chance": chance}


class LinearProbeAnalyzer(Analyzer):
    name = "linear_probe"

    def run(self, representations: dict[str, np.ndarray], metadata: dict[str, np.ndarray], output_dir: Path) -> dict[str, Any]:
        output_dir.mkdir(parents=True, exist_ok=True)
        groups = metadata.get("episode")
        factors, skipped = {}, {}
        for name, (target, kind) in factor_columns(metadata).items():
            reason = skip_reason(target, kind, groups)
            if reason is None:
                factors[name] = (target, kind)
            else:
                skipped[name] = reason

        rows = []
        for rep_name, values in representations.items():
            for factor, (target, kind) in factors.items():
                result = probe(values, target, kind, groups)
                rows.append({"representation": rep_name, "factor": factor, "kind": kind, **result})

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
