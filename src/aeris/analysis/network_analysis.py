"""Analyses of a layer's activations: spectrum, probes, clusters and alignment.

Every function takes plain arrays (one row per state) and needs only NumPy;
nothing here knows about agents, environments or files. With few states
(68 on Four Rooms) and more units than states, a probe can fit any target on
its own training data, so every probe score here is measured on held-out
states (k-fold), with its regularization chosen inside each training fold.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ALPHAS = np.logspace(-3, 5, 17)  # ridge strengths tried, relative to the mean eigenvalue
PROBES = ("linear", "kernel", "knn")


# Spectrum ------------------------------------------------------------------------------


def centered(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x - x.mean(axis=0, keepdims=True)


def singular_values(x: np.ndarray) -> np.ndarray:
    """Singular values of the centered activations, largest first."""
    return np.linalg.svd(centered(x), compute_uv=False)


def effective_rank(singular: np.ndarray) -> float:
    """exp(entropy) of the normalized variance spectrum (Roy & Vetterli); 0 if constant."""
    variance = np.asarray(singular, dtype=np.float64) ** 2
    if variance.sum() <= 1e-24:
        return 0.0
    p = variance[variance > 0] / variance.sum()
    return float(np.exp(-(p * np.log(p)).sum()))


def srank(singular: np.ndarray, delta: float = 0.01) -> int:
    """Fewest singular values holding 1 − delta of their sum (Kumar et al. 2021)."""
    singular = np.asarray(singular, dtype=np.float64)
    if singular.sum() <= 1e-12:
        return 0
    cumulative = np.cumsum(singular) / singular.sum()
    return int(np.searchsorted(cumulative, 1 - delta) + 1)


def pca(x: np.ndarray, components: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Principal-component scores [N, c] and the share of variance of each component."""
    xc = centered(x)
    u, singular, _ = np.linalg.svd(xc, full_matrices=False)
    variance = singular**2
    total = variance.sum()
    c = min(components, len(singular))
    share = variance[:c] / total if total > 0 else np.zeros(c)
    return u[:, :c] * singular[:c], share


def dead_units(x: np.ndarray, tolerance: float = 1e-8) -> float:
    """Share of units that never change across states (e.g. ReLUs that never fire)."""
    x = np.asarray(x, dtype=np.float64)
    return float(np.mean(x.std(axis=0) <= tolerance)) if x.shape[1] else 0.0


# Probes --------------------------------------------------------------------------------


@dataclass(frozen=True)
class ProbeResult:
    """A held-out probe score: R² (continuous) or accuracy (categorical)."""

    score: float
    chance: float  # the best constant prediction: R² 0, or the most accepted class
    predictions: np.ndarray  # held-out prediction of every state: values [N, m] or classes [N]
    metric: str  # "R²" or "accuracy"


def folds(n: int, count: int = 5, seed: int = 0) -> list[np.ndarray]:
    """Test indices of ``count`` shuffled folds covering 0..n-1 once each."""
    order = np.random.default_rng(seed).permutation(n)
    return [np.sort(part) for part in np.array_split(order, min(count, n))]


def _standardize(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Center on the training fold and divide by one overall scale, keeping the layer's own
    geometry: rescaling each unit would blow up units that barely vary."""
    mean = train.mean(axis=0)
    scale = float(np.sqrt(((train - mean) ** 2).mean()))
    scale = scale if scale > 1e-12 else 1.0
    return (train - mean) / scale, (test - mean) / scale


def _ridge(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray) -> np.ndarray:
    """Ridge with intercept, its strength picked by exact leave-one-out on the training fold."""
    x_mean, y_mean = train_x.mean(axis=0), train_y.mean(axis=0)
    xc, yc = train_x - x_mean, train_y - y_mean
    u, singular, vt = np.linalg.svd(xc, full_matrices=False)
    variance = singular**2
    scale = variance.mean() if variance.size and variance.mean() > 0 else 1.0
    uy = u.T @ yc
    n = len(train_x)
    best, best_error = ALPHAS[0] * scale, np.inf
    for alpha in ALPHAS * scale:
        shrink = variance / (variance + alpha)
        fitted = u @ (shrink[:, None] * uy)
        leverage = (u**2 @ shrink) + 1 / n
        error = np.sum(((yc - fitted) / np.maximum(1 - leverage, 1e-6)[:, None]) ** 2)
        if error < best_error:
            best, best_error = alpha, error
    coefficients = vt.T @ ((singular / (variance + best))[:, None] * uy)
    return (test_x - x_mean) @ coefficients + y_mean


def _kernel_ridge(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray) -> np.ndarray:
    """RBF kernel ridge (bandwidth: median squared distance), strength by leave-one-out."""

    def squared(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.maximum((a**2).sum(1)[:, None] + (b**2).sum(1)[None, :] - 2 * a @ b.T, 0.0)

    distances = squared(train_x, train_x)
    width = np.median(distances[distances > 0]) if np.any(distances > 0) else 1.0
    kernel = np.exp(-distances / width)
    y_mean = train_y.mean(axis=0)
    yc = train_y - y_mean
    eigenvalues, vectors = np.linalg.eigh(kernel)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    vy = vectors.T @ yc
    scale = eigenvalues.mean() if eigenvalues.mean() > 0 else 1.0
    best, best_error = ALPHAS[0] * scale, np.inf
    for alpha in ALPHAS * scale:
        shrink = eigenvalues / (eigenvalues + alpha)
        fitted = vectors @ (shrink[:, None] * vy)
        leverage = vectors**2 @ shrink
        error = np.sum(((yc - fitted) / np.maximum(1 - leverage, 1e-6)[:, None]) ** 2)
        if error < best_error:
            best, best_error = alpha, error
    weights = vectors @ (vy / (eigenvalues + best)[:, None])
    return np.exp(-squared(test_x, train_x) / width) @ weights + y_mean


def _knn(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, k: int = 5) -> np.ndarray:
    """Mean of the k nearest training states' targets (Euclidean, standardized units)."""
    distances = ((test_x[:, None, :] - train_x[None, :, :]) ** 2).sum(-1)
    nearest = np.argsort(distances, axis=1)[:, : min(k, len(train_x))]
    return train_y[nearest].mean(axis=1)


_FITS = {"linear": _ridge, "kernel": _kernel_ridge, "knn": _knn}


def probe(
    x: np.ndarray,
    y: np.ndarray,
    *,
    kind: str = "linear",
    categorical: bool = False,
    accepted: np.ndarray | None = None,
    fold_count: int = 5,
    seed: int = 0,
) -> ProbeResult:
    """Held-out score of predicting ``y`` from ``x`` with a linear, kernel or kNN probe.

    Continuous ``y`` ([N] or [N, m]) gives R² pooled over outputs. Categorical
    ``y`` ([N] class indices) is fit on one-hot targets and predicted by the
    largest output; ``accepted`` [N, classes] marks every class that counts as
    correct for a state (e.g. any optimal action), by default only ``y`` itself.
    """
    if kind not in _FITS:
        raise ValueError(f"unknown probe {kind!r}; choose from {PROBES}")
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y)
    n = len(x)
    if categorical:
        classes = int(np.max(y)) + 1
        targets = np.eye(classes)[y.astype(int)]
        accepted = targets.astype(bool) if accepted is None else np.asarray(accepted, dtype=bool)
    else:
        targets = y.astype(np.float64).reshape(n, -1)
    predicted = np.zeros_like(targets, dtype=np.float64)
    for test in folds(n, fold_count, seed):
        train = np.setdiff1d(np.arange(n), test)
        train_x, test_x = _standardize(x[train], x[test])
        predicted[test] = _FITS[kind](train_x, targets[train], test_x)
    if categorical:
        assert accepted is not None
        chosen = predicted.argmax(axis=1)
        score = float(accepted[np.arange(n), chosen].mean())
        chance = float(np.max(accepted.mean(axis=0)))
        return ProbeResult(score, chance, chosen, "accuracy")
    total = ((targets - targets.mean(axis=0)) ** 2).sum()
    score = float(1 - ((targets - predicted) ** 2).sum() / total) if total > 0 else 0.0
    return ProbeResult(score, 0.0, predicted.squeeze(1) if y.ndim == 1 else predicted, "R²")


# Clusters ------------------------------------------------------------------------------


def kmeans(x: np.ndarray, k: int, *, restarts: int = 10, seed: int = 0) -> np.ndarray:
    """k-means++ with Lloyd iterations; the labelling with the lowest inertia of the restarts."""
    x = centered(x)
    rng = np.random.default_rng(seed)
    k = min(k, len(x))
    best, best_inertia = np.zeros(len(x), dtype=int), np.inf
    for _ in range(restarts):
        centres = [x[rng.integers(len(x))]]
        for _ in range(1, k):
            distance = np.min([((x - c) ** 2).sum(1) for c in centres], axis=0)
            total = distance.sum()
            pick = rng.choice(len(x), p=distance / total) if total > 0 else rng.integers(len(x))
            centres.append(x[pick])
        centre = np.array(centres)
        for _ in range(100):
            labels = ((x[:, None, :] - centre[None]) ** 2).sum(-1).argmin(1)
            updated = np.array(
                [x[labels == j].mean(0) if np.any(labels == j) else centre[j] for j in range(k)]
            )
            if np.allclose(updated, centre):
                break
            centre = updated
        inertia = ((x - centre[labels]) ** 2).sum()
        if inertia < best_inertia:
            best, best_inertia = labels, inertia
    return _relabel(best)


def ward(x: np.ndarray, k: int) -> np.ndarray:
    """Agglomerative clustering with Ward linkage, cut at ``k`` clusters."""
    x = centered(x)
    clusters = [[i] for i in range(len(x))]
    sizes = np.ones(len(x))
    centres = x.copy()
    while len(clusters) > max(k, 1):
        best, pair = np.inf, (0, 1)
        for i in range(len(clusters)):
            gap = ((centres[i + 1 :] - centres[i]) ** 2).sum(1)
            cost = sizes[i] * sizes[i + 1 :] / (sizes[i] + sizes[i + 1 :]) * gap
            if cost.size and cost.min() < best:
                best, pair = cost.min(), (i, i + 1 + int(cost.argmin()))
        i, j = pair
        total = sizes[i] + sizes[j]
        centres[i] = (sizes[i] * centres[i] + sizes[j] * centres[j]) / total
        sizes[i] = total
        clusters[i] += clusters[j]
        del clusters[j]
        centres = np.delete(centres, j, axis=0)
        sizes = np.delete(sizes, j)
    labels = np.zeros(len(x), dtype=int)
    for index, members in enumerate(clusters):
        labels[members] = index
    return _relabel(labels)


def _relabel(labels: np.ndarray) -> np.ndarray:
    """Number clusters in order of first appearance, so the colours are stable."""
    order = {label: index for index, label in enumerate(dict.fromkeys(labels.tolist()))}
    return np.array([order[label] for label in labels.tolist()], dtype=int)


def adjusted_rand(a: np.ndarray, b: np.ndarray) -> float:
    """Adjusted Rand index of two labellings: 1 identical, about 0 unrelated."""
    first = np.unique(a, return_inverse=True)[1].ravel()
    second = np.unique(b, return_inverse=True)[1].ravel()
    table = np.zeros((int(np.max(first)) + 1, int(np.max(second)) + 1))
    np.add.at(table, (first, second), 1)

    def pairs(values: np.ndarray) -> float:
        return float((values * (values - 1) / 2).sum())

    n = len(first)
    index, rows, columns = pairs(table), pairs(table.sum(1)), pairs(table.sum(0))
    expected = rows * columns / (n * (n - 1) / 2)
    maximum = (rows + columns) / 2
    return 1.0 if maximum == expected else (index - expected) / (maximum - expected)


# Alignment and sensitivity -------------------------------------------------------------


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    """Linear centered kernel alignment (Kornblith et al. 2019): 1 for the same geometry up
    to rotation and scale; not invariant to other invertible linear maps."""
    xc, yc = centered(x), centered(y)
    cross = np.linalg.norm(yc.T @ xc) ** 2
    norms = np.linalg.norm(xc.T @ xc) * np.linalg.norm(yc.T @ yc)
    return float(cross / norms) if norms > 0 else 0.0


def relative_change(x: np.ndarray, reference: np.ndarray) -> float:
    """‖x − reference‖ / ‖reference‖ over all states and units: 0 unchanged; sees scale too,
    unlike CKA."""
    x, reference = np.asarray(x, dtype=np.float64), np.asarray(reference, dtype=np.float64)
    norm = np.linalg.norm(reference)
    return float(np.linalg.norm(x - reference) / norm) if norm > 0 else 0.0


def edge_distances(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Feature distance across each edge [E, 2], over the median distance of all state pairs."""
    x = np.asarray(x, dtype=np.float64)
    gaps = np.linalg.norm(x[edges[:, 0]] - x[edges[:, 1]], axis=1)
    all_pairs = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=-1)
    median = np.median(all_pairs[np.triu_indices(len(x), 1)])
    return gaps / median if median > 0 else gaps
