"""Calibration diagnostics for production probability monitoring."""
from __future__ import annotations

import numpy as np


def reliability(p: np.ndarray, y: np.ndarray, n_bins: int = 10
                ) -> tuple[np.ndarray, np.ndarray]:
    sorted_idx = np.argsort(p)
    p_sorted = p[sorted_idx]
    y_sorted = y[sorted_idx]
    bin_edges = np.array_split(np.arange(len(p)), n_bins)
    bin_centres = np.array([p_sorted[idx].mean() for idx in bin_edges if len(idx) > 0])
    mean_y = np.array([y_sorted[idx].mean() for idx in bin_edges if len(idx) > 0])
    return bin_centres, mean_y


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))
