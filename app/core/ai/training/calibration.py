"""Probability calibration.

Why we always calibrate: Kelly sizing and the meta-labeler threshold both
require probabilities to be honest. LightGBM's raw outputs are not.

Use isotonic on the held-out fold of the walk-forward split; report
reliability + Brier on the next fold.
"""
from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression


class IsotonicCalibrator:
    def __init__(self) -> None:
        self._model: IsotonicRegression | None = None

    def fit(self, p_raw: np.ndarray, y: np.ndarray) -> None:
        self._model = IsotonicRegression(out_of_bounds="clip")
        self._model.fit(p_raw, y)

    def transform(self, p_raw: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("IsotonicCalibrator must be fit before transform.")
        return self._model.predict(p_raw).astype(np.float64)


def reliability_curve(p: np.ndarray, y: np.ndarray, n_bins: int = 10
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Returns (mean predicted prob per bin, observed positive rate per bin)."""
    # Equal-frequency bins
    sorted_idx = np.argsort(p)
    p_sorted = p[sorted_idx]
    y_sorted = y[sorted_idx]
    bin_edges = np.array_split(np.arange(len(p)), n_bins)
    bin_centres = np.array([p_sorted[idx].mean() for idx in bin_edges if len(idx) > 0])
    mean_y = np.array([y_sorted[idx].mean() for idx in bin_edges if len(idx) > 0])
    return bin_centres, mean_y


def brier_score(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))
