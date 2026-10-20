"""Linear CKA between representations of the same samples (Kornblith et al., 2019).

CKA compares how two representations arrange the samples, not their coordinates: it is invariant to rotations and
isotropic scaling of either representation, so representations from unrelated networks and of different widths can
be compared. Linear CKA is biased upwards for wide representations relative to the number of samples; read values
against control representations (untrained networks, raw observations) in the same matrix.
"""

from __future__ import annotations

import numpy as np


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    """CKA with linear kernels, computed in feature space: ||Yc^T Xc||_F^2 / (||Xc^T Xc||_F ||Yc^T Yc||_F)."""
    x = x - x.mean(axis=0)
    y = y - y.mean(axis=0)
    cross = np.linalg.norm(y.T @ x) ** 2
    norm = np.linalg.norm(x.T @ x) * np.linalg.norm(y.T @ y)
    return float(cross / norm) if norm > 0 else 0.0


def cka_matrix(representations: dict[str, np.ndarray]) -> tuple[list[str], np.ndarray]:
    """Symmetric matrix of linear CKA between every pair of representations (rows in the same sample order)."""
    names = list(representations)
    matrix = np.eye(len(names))
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            matrix[i, j] = matrix[j, i] = linear_cka(representations[names[i]], representations[names[j]])
    return names, matrix
