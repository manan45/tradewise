"""Pull bars (equities + futures) and persist to TimescaleDB."""
from __future__ import annotations

from datetime import datetime, timedelta

from app.core.ports.market_data import MarketDataProvider


async def run(
    asof: datetime,
    provider: MarketDataProvider,
    symbols: list[str],
    interval: str = "1d",
    db=None,
) -> int:
    start = asof - timedelta(days=1)
    rows_written = 0
    for symbol in symbols:
        bars = await provider.get_bars(symbol, start, asof, interval)
        for bar in bars:
            # Validate OHLC sanity
            if bar.high < bar.open or bar.high < bar.close:
                continue
            if bar.low > bar.open or bar.low > bar.close:
                continue
            if db is not None:
                await db.execute(
                    """
                    INSERT INTO prices_eod (symbol, ts, open, high, low, close, volume, interval, asof_ingested_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, NOW())
                    ON CONFLICT (symbol, ts) DO UPDATE
                    SET open=$3, high=$4, low=$5, close=$6, volume=$7, asof_ingested_at=NOW()
                    """,
                    symbol, bar.ts, bar.open, bar.high, bar.low, bar.close, bar.volume, interval,
                )
            rows_written += 1
    return rows_written
