"""Calibration metrics shared by P2 training and evaluation.

All metric functions take ``y`` (binary 0/1 labels) and ``p`` (a score or
probability array). ECE, Brier and mean-confidence-on-wrong only make sense
for scores that are genuine probabilities in [0, 1]; callers should pass them
only for probability-valued methods (Prob/Consis/Verbal confidence, temperature
scaling, Platt, isotonic, MLP outputs). AUROC and Alignment are rank-based and
therefore valid for any real-valued score.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def ece(y: np.ndarray, p: np.ndarray, n_bins: int = 15) -> float:
    """Expected calibration error with equal-width bins over [0, 1]."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(y)
    value = 0.0
    for i in range(n_bins):
        sel = (p >= edges[i]) & (p <= edges[i + 1])
        if sel.sum() > 0:
            value += (sel.sum() / total) * abs(p[sel].mean() - y[sel].mean())
    return float(value)


def brier(y: np.ndarray, p: np.ndarray) -> float:
    """Brier score: mean squared error between probability and label."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    return float(np.mean((y - p) ** 2))


def auroc(y: np.ndarray, p: np.ndarray) -> float:
    """Area under the ROC curve; NaN when the test set has a single class."""
    y = np.asarray(y)
    p = np.asarray(p, dtype=float)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def conf_wrong(y: np.ndarray, p: np.ndarray) -> float:
    """Mean score assigned to incorrect (y=0) samples, the paper's Confw."""
    y = np.asarray(y)
    p = np.asarray(p, dtype=float)
    if (y == 0).sum() == 0:
        return float("nan")
    return float(p[y == 0].mean())


def select_threshold(y: np.ndarray, p: np.ndarray) -> float:
    """Pick the threshold maximizing accuracy, scanning unique scores.

    Matches the paper's appendix B.2 rule for confidence-valued methods:
    predict correct when ``score >= threshold``.
    """
    y = np.asarray(y)
    p = np.asarray(p, dtype=float)
    best_thr, best_acc = 0.5, -1.0
    for thr in np.unique(p):
        acc = float(np.mean((p >= thr) == y))
        if acc > best_acc:
            best_acc, best_thr = acc, float(thr)
    return best_thr


def alignment(y: np.ndarray, p: np.ndarray, threshold: float | None = None) -> float:
    """Accuracy of the thresholded score. If no threshold, select it on the data."""
    y = np.asarray(y)
    p = np.asarray(p, dtype=float)
    if threshold is None:
        threshold = select_threshold(y, p)
    return float(np.mean((p >= threshold) == y))
