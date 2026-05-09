"""FRED macro series ingest (DGS10, DGS2, T10Y2Y, DFF, DXY, CPI, ...)."""
from __future__ import annotations

from datetime import date, timedelta

from app.core.ports.macro import MacroDataProvider


DEFAULT_SERIES = (
    "DGS10", "DGS2", "T10Y2Y", "DFF", "DTWEXBGS",
    "CPIAUCSL", "PPIACO", "UNRATE", "PAYEMS",
)


async def run(asof: date, provider: MacroDataProvider,
              series: tuple[str, ...] = DEFAULT_SERIES,
              db=None) -> int:
    start = asof - timedelta(days=7)
    rows_written = 0
    for series_id in series:
        observations = await provider.get_series(series_id, start, asof)
        for obs_date, value in observations:
            if db is not None:
                await db.execute(
                    """
                    INSERT INTO macro_series (series_id, date, value, asof_ingested_at)
                    VALUES ($1, $2, $3, NOW())
                    ON CONFLICT (series_id, date) DO UPDATE SET value=$3, asof_ingested_at=NOW()
                    """,
                    series_id, obs_date, value,
                )
            rows_written += 1
    return rows_written
