"""Point-in-time helpers — the only sanctioned way to read past data.

Wraps polars.DataFrame.join_asof so callers can't accidentally use look-ahead.
Every feature pipeline must go through these helpers.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

import polars as pl


def asof_join(
    left: pl.DataFrame,
    right: pl.DataFrame,
    on: str = "ts",
    by: str | None = "symbol",
    tolerance: str | None = None,
) -> pl.DataFrame:
    """Strict backward asof join — for every row in `left`, attach the latest
    row in `right` whose `on` <= left.on. ``tolerance`` is a polars duration
    string (e.g. '5m') beyond which no match is attached."""
    left_sorted = left.sort(on)
    right_sorted = right.sort(on)
    kwargs: dict[str, Any] = {"on": on, "strategy": "backward"}
    if by is not None:
        kwargs["by"] = by
    if tolerance is not None:
        kwargs["tolerance"] = tolerance
    return left_sorted.join_asof(right_sorted, **kwargs)


def asof_lookup(
    series: pl.DataFrame,
    ts: datetime,
    column: str,
    by_value: str | None = None,
    by_column: str | None = "symbol",
) -> Any:
    """Single-point asof lookup — returns the value of `column` at the largest
    timestamp <= `ts` (optionally filtered by `by_column == by_value`)."""
    df = series
    if by_column is not None and by_value is not None:
        df = df.filter(pl.col(by_column) == by_value)
    df = df.filter(pl.col("ts") <= ts).sort("ts")
    if df.is_empty():
        return None
    return df[-1][column][0]
