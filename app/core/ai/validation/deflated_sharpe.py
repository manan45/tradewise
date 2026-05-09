"""Deflated Sharpe wrapper that the dashboard's Backtest Viewer consumes.

Imported separately from the training metric so dashboard code doesn't pull
in mlfinlab + lightgbm at import time.
"""
from __future__ import annotations

from app.core.ai.training.metrics import deflated_sharpe as _deflated_sharpe


def deflated_sharpe(
    sharpe_estimate: float,
    n_trials: int,
    skew: float,
    kurtosis: float,
    n_obs: int,
) -> float:
    return _deflated_sharpe(sharpe_estimate, n_trials, skew, kurtosis, n_obs)
