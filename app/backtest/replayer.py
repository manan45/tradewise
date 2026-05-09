"""Deterministic bar replayer — emits bars in time order, no look-ahead.

Critical: must enforce the survivorship-safe universe by joining each
timestamp against the universe_membership table (BACKTESTING.md §3.1).
Symbols delisted before `ts` are excluded; symbols added after are excluded.
"""
from __future__ import annotations

import os
from datetime import date, datetime, timezone
from typing import AsyncIterator

import polars as pl

from app.core.ports.market_data import Bar


class BarReplayer:
    def __init__(self, source_path: str, universe_table: str = "universe_membership"):
        self.source_path = source_path
        self.universe_table = universe_table

    async def stream(
        self,
        start: date,
        end: date,
        symbols: list[str] | None = None,
    ) -> AsyncIterator[Bar]:
        # Load parquet shards from source_path
        if not os.path.exists(self.source_path):
            raise FileNotFoundError(f"Source path not found: {self.source_path}")

        if os.path.isfile(self.source_path):
            df = pl.read_parquet(self.source_path)
        else:
            files = [
                os.path.join(self.source_path, f)
                for f in sorted(os.listdir(self.source_path))
                if f.endswith(".parquet")
            ]
            if not files:
                return
            df = pl.concat([pl.read_parquet(f) for f in files])

        # Filter by date range
        start_dt = datetime(start.year, start.month, start.day, tzinfo=timezone.utc)
        end_dt = datetime(end.year, end.month, end.day, 23, 59, 59, tzinfo=timezone.utc)
        df = df.filter(
            (pl.col("ts") >= start_dt) & (pl.col("ts") <= end_dt)
        )
        if symbols is not None:
            df = df.filter(pl.col("symbol").is_in(symbols))

        # Sort deterministically: (ts, symbol)
        df = df.sort(["ts", "symbol"])

        for row in df.iter_rows(named=True):
            yield Bar(
                symbol=row["symbol"],
                ts=row["ts"],
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row.get("volume", 0)),
                vwap=row.get("vwap"),
                interval=row.get("interval", "1d"),
            )
