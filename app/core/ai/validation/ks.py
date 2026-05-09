"""Two-sample Kolmogorov-Smirnov for distributional drift."""
from __future__ import annotations

import numpy as np
from scipy import stats


def ks_test(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Returns (statistic, p_value)."""
    result = stats.ks_2samp(a, b)
    return float(result.statistic), float(result.pvalue)
