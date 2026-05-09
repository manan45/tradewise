"""Performance metrics — finance-aware, multiple-testing-aware."""
from __future__ import annotations

import math

import numpy as np
from scipy import stats


def sharpe(returns: np.ndarray, periods_per_year: int = 252) -> float:
    std = np.std(returns, ddof=1)
    if std == 0:
        return 0.0
    return float(np.mean(returns) / std * math.sqrt(periods_per_year))


def deflated_sharpe(
    sharpe_estimate: float,
    n_trials: int,
    skew: float,
    kurtosis: float,
    n_obs: int,
) -> float:
    """Bailey & Lopez de Prado deflated Sharpe — strips multiple-testing bias."""
    # Expected maximum Sharpe under H0 (standard normal order statistics approximation)
    euler_mascheroni = 0.5772156649
    # E[max SR] ≈ (1 - γ) * Z^{-1}(1 - 1/n_trials) + γ * Z^{-1}(1 - 1/(n_trials * e))
    e_max_sr = (
        (1 - euler_mascheroni) * stats.norm.ppf(1 - 1.0 / n_trials)
        + euler_mascheroni * stats.norm.ppf(1 - 1.0 / (n_trials * math.e))
    ) if n_trials > 1 else 0.0

    # Variance correction for non-normality
    var_sr = (1 - skew * sharpe_estimate + ((kurtosis - 1) / 4) * sharpe_estimate ** 2) / (n_obs - 1)
    sigma_sr = math.sqrt(var_sr) if var_sr > 0 else 1e-9

    z = (sharpe_estimate - e_max_sr) / sigma_sr
    return float(stats.norm.cdf(z))


def probabilistic_sharpe_ratio(
    sharpe_estimate: float,
    benchmark: float,
    skew: float,
    kurtosis: float,
    n_obs: int,
) -> float:
    var_sr = (1 - skew * sharpe_estimate + ((kurtosis - 1) / 4) * sharpe_estimate ** 2) / (n_obs - 1)
    sigma_sr = math.sqrt(var_sr) if var_sr > 0 else 1e-9
    z = (sharpe_estimate - benchmark) / sigma_sr
    return float(stats.norm.cdf(z))


def max_drawdown(equity: np.ndarray) -> float:
    running_max = np.maximum.accumulate(equity)
    drawdowns = (equity - running_max) / running_max
    return float(np.min(drawdowns))


def hit_rate(returns: np.ndarray) -> float:
    return float(np.mean(returns > 0))


def profit_factor(returns: np.ndarray) -> float:
    gains = returns[returns > 0]
    losses = returns[returns < 0]
    if len(losses) == 0:
        return float("inf")
    return float(np.sum(gains) / abs(np.sum(losses)))