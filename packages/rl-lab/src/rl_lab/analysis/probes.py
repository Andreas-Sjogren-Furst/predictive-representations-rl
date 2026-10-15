"""Linear probes: how well a linear readout of a representation recovers each ground-truth factor.

Continuous factors: ridge regression, cross-validated R^2. Categorical factors (rooms, rewarded-or-not): class-balanced
logistic regression, balanced accuracy, so chance is 1/k however imbalanced the classes are. Folds are split by group
(e.g. episode) when groups are given, because overlapping samples from one episode would otherwise leak between train
and test; categorical folds are also stratified by class. Standardisation is fitted on the training folds only.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.model_selection import GroupKFold, KFold, StratifiedGroupKFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

NUM_FOLDS = 5
# A class needs this many samples, from at least NUM_FOLDS groups, to be probed; otherwise the factor is skipped.
MIN_CLASS_SIZE = 20
RIDGE_ALPHAS = np.logspace(-3, 3, 13)

# Bookkeeping columns of a sample table that are not properties of the sample.
NON_FACTORS = frozenset({"source_index", "episode"})
# Integer labels with at most this many distinct values are treated as classes.
MAX_CLASSES = 20
# `reward_<task>` labels are probed as "rewarded or not". Tolerance-style rewards decay smoothly and only underflow to
# exactly 0 far from the goal, so a tiny threshold separates near-goal states from the rest.
REWARD_THRESHOLD = 1e-6


def factor_columns(metadata: dict[str, np.ndarray]) -> dict[str, tuple[np.ndarray, str]]:
    """Label columns to probe for, each tagged 'categorical' or 'continuous'.

    `reward_<task>` columns become binary `rewarded_<task>` labels (reward > REWARD_THRESHOLD): rewards are mostly 0,
    which makes regression R^2 meaningless. Ordered: environment factors, then rewards, then the step in the episode.
    """
    def order(key: str) -> int:
        return 2 if key == "step" else 1 if key.startswith("reward_") else 0

    factors = {}
    for key, values in sorted(metadata.items(), key=lambda item: order(item[0])):
        if key in NON_FACTORS or values.ndim != 1:
            continue
        if key.startswith("reward_"):
            factors["rewarded_" + key.removeprefix("reward_")] = ((values > REWARD_THRESHOLD).astype(np.int64), "categorical")
            continue
        categorical = np.issubdtype(values.dtype, np.integer) and len(np.unique(values)) <= MAX_CLASSES
        factors[key] = (values, "categorical" if categorical else "continuous")
    return factors


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
    """Cross-validated linear probe of one factor: metric, mean score, std over folds and chance level."""
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


def probe_table(
    representations: dict[str, np.ndarray], metadata: dict[str, np.ndarray], groups: np.ndarray | None
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Probe every factor from every representation. Returns one row per (representation, factor), and the factors
    that were skipped with the reason."""
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
            rows.append({"representation": rep_name, "factor": factor, "kind": kind, **probe(values, target, kind, groups)})
    return rows, skipped
