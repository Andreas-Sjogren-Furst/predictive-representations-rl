"""PCA and effective dimensionality of a representation."""

from __future__ import annotations

from typing import Any

import numpy as np

NUM_SUMMARY_COMPONENTS = 10


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


def pca_summary(values: np.ndarray) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Summary statistics of a representation's spectrum, with its PCA scores and explained-variance ratios."""
    scores, ratio = pca(values)
    cumulative = np.cumsum(ratio)
    summary = {
        "dims": int(values.shape[1]),
        "components_for_90pct": int(np.searchsorted(cumulative, 0.90) + 1),
        "components_for_95pct": int(np.searchsorted(cumulative, 0.95) + 1),
        "participation_ratio": round(participation_ratio(ratio), 2),
        "explained_variance_ratio_top10": [round(float(r), 4) for r in ratio[:NUM_SUMMARY_COMPONENTS]],
    }
    return summary, scores, ratio
