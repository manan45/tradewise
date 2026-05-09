"""Population Stability Index — feature drift between two distributions.

Rule of thumb: PSI < 0.1 stable; 0.1-0.25 minor drift; >0.25 retrain.
"""
from __future__ import annotations

import numpy as np


def psi_categorical(expected: np.ndarray, actual: np.ndarray) -> float:
    cats = np.union1d(expected, actual)
    eps = 1e-9
    psi = 0.0
    n_exp = len(expected)
    n_act = len(actual)
    for c in cats:
        p_e = np.sum(expected == c) / n_exp + eps
        p_a = np.sum(actual == c) / n_act + eps
        psi += (p_a - p_e) * np.log(p_a / p_e)
    return float(psi)


def psi_continuous(expected: np.ndarray, actual: np.ndarray,
                   n_bins: int = 10) -> float:
    eps = 1e-9
    breakpoints = np.percentile(expected, np.linspace(0, 100, n_bins + 1))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf
    psi = 0.0
    for i in range(n_bins):
        p_e = np.mean((expected >= breakpoints[i]) & (expected < breakpoints[i + 1])) + eps
        p_a = np.mean((actual >= breakpoints[i]) & (actual < breakpoints[i + 1])) + eps
        psi += (p_a - p_e) * np.log(p_a / p_e)
    return float(psi)
