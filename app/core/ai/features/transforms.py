"""Building-block transforms reused across features.

Each function takes a polars Series (or DataFrame) and returns a new Series.
Keep them pure, deterministic, and side-effect free so they hash stably for
the registry. NaN handling: never fill — propagate NaN downstream so we can
tell missing data from real zeros.
"""
from __future__ import annotations

import math

import polars as pl


def log_return(close: pl.Series, periods: int = 1) -> pl.Series:
    return (close / close.shift(periods)).log(math.e).alias("log_return")


def realized_vol(close: pl.Series, window: int) -> pl.Series:
    """Annualized realized vol from log returns over `window` bars."""
    lr = log_return(close)
    return (lr.rolling_std(window_size=window) * math.sqrt(252)).alias("realized_vol")


def parkinson_vol(high: pl.Series, low: pl.Series, window: int) -> pl.Series:
    """High-low range estimator (~5x more efficient than close-to-close)."""
    log_hl_sq = (high / low).log(math.e) ** 2
    factor = 1.0 / (4.0 * math.log(2))
    rv = (log_hl_sq.rolling_mean(window_size=window) * factor).sqrt() * math.sqrt(252)
    return rv.alias("parkinson_vol")


def atr(
    high: pl.Series,
    low: pl.Series,
    close: pl.Series,
    window: int = 14,
) -> pl.Series:
    prev_close = close.shift(1)
    hl = high - low
    hpc = (high - prev_close).abs()
    lpc = (low - prev_close).abs()
    tr_vals = [
        max(h, hp, lp) if (h is not None and hp is not None and lp is not None) else None
        for h, hp, lp in zip(hl.to_list(), hpc.to_list(), lpc.to_list())
    ]
    tr = pl.Series("_tr", tr_vals, dtype=pl.Float64)
    return tr.ewm_mean(span=window).alias("atr")


def rsi(close: pl.Series, window: int = 14) -> pl.Series:
    delta = close - close.shift(1)
    gain = delta.clip(lower_bound=0.0)
    loss = (-delta).clip(lower_bound=0.0)
    avg_gain = gain.ewm_mean(alpha=1.0 / window, adjust=False)
    avg_loss = loss.ewm_mean(alpha=1.0 / window, adjust=False)
    rs = avg_gain / avg_loss
    return (100.0 - (100.0 / (1.0 + rs))).alias("rsi")


def zscore(s: pl.Series, window: int) -> pl.Series:
    mean = s.rolling_mean(window_size=window)
    std = s.rolling_std(window_size=window)
    return ((s - mean) / std).alias("zscore")


def rolling_rank_pct(s: pl.Series, window: int) -> pl.Series:
    """Where in the rolling distribution does the latest value sit (0..1)."""
    vals = s.to_list()
    n = len(vals)
    result: list[float | None] = [None] * n
    for i in range(window - 1, n):
        window_vals = [v for v in vals[i - window + 1 : i + 1] if v is not None]
        current = vals[i]
        if not window_vals or current is None:
            continue
        result[i] = sum(1 for v in window_vals if v <= current) / len(window_vals)
    return pl.Series(s.name or "rolling_rank_pct", result, dtype=pl.Float64)


def cross_sectional_rank(df: pl.DataFrame, value_col: str,
                         group_col: str = "ts") -> pl.Series:
    """Rank `value_col` within each `group_col` bucket (per-bar ranking)."""
    ranked = df.with_columns(
        pl.col(value_col).rank(method="average").over(group_col).alias("_rank"),
        pl.col(value_col).count().over(group_col).alias("_count"),
    ).with_columns(
        (pl.col("_rank") / pl.col("_count")).alias("_pct_rank")
    )
    return ranked["_pct_rank"].alias(f"{value_col}_cs_rank")
