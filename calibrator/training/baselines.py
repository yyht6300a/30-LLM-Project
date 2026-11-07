"""Calibration baselines: thresholded scores and monotonic recalibration.

ThresholdModel fits a single decision threshold on a one-column score and may
flip the comparison direction, so features that correlate negatively with
correctness (for example generated-answer popularity) still get a sensible
threshold. TemperatureScaler / PlattScaler / IsotonicCalibrator map a raw
confidence to a recalibrated probability in [0, 1].
"""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


class ThresholdModel:
    """Binarize a one-column score; direction and threshold fit on training data."""

    def __init__(self) -> None:
        self.threshold: float = 0.5
        self.flip: bool = False

    def fit(self, scores: np.ndarray, y: np.ndarray) -> "ThresholdModel":
        scores = np.asarray(scores, dtype=float)
        y = np.asarray(y)
        best_acc = -1.0
        for thr in np.unique(scores):
            for flip in (False, True):
                pred = (scores <= thr) if flip else (scores >= thr)
                acc = float(np.mean(pred == y))
                if acc > best_acc:
                    best_acc, self.threshold, self.flip = acc, float(thr), flip
        return self

    def predict(self, scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(scores, dtype=float)
        return (scores <= self.threshold) if self.flip else (scores >= self.threshold)


class TemperatureScaler:
    """Single-parameter temperature scaling over the confidence log-odds.

    Fitted with logistic regression constrained to zero intercept, so the
    learned coefficient is ``1 / T``. Monotonic in the input confidence.
    """

    def __init__(self) -> None:
        self.w_ = 1.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TemperatureScaler":
        z = _logit(X[:, 0]).reshape(-1, 1)
        model = LogisticRegression(fit_intercept=False, max_iter=1000)
        model.fit(z, y)
        self.w_ = float(model.coef_[0, 0])
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        z = _logit(X[:, 0]) * self.w_
        return 1.0 / (1.0 + np.exp(-z))


class PlattScaler:
    """Logistic regression over the raw confidence score (Platt-style)."""

    def __init__(self) -> None:
        self._model = LogisticRegression(max_iter=1000)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PlattScaler":
        self._model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X)[:, 1]


class IsotonicCalibrator:
    """Isotonic regression mapping confidence to a monotone probability."""

    def __init__(self) -> None:
        self._model = IsotonicRegression(out_of_bounds="clip")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "IsotonicCalibrator":
        self._model.fit(X[:, 0], y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X[:, 0])
