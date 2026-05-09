"""Earnings-calendar ingest — populates `earnings_calendar` for blackouts."""
from __future__ import annotations

from datetime import date, timedelta


async def run(asof: date, lookahead_days: int = 14,
              provider=None, db=None) -> int:
    """Fetch next `lookahead_days` earnings events and upsert into earnings_calendar."""
    if provider is None:
        return 0

    end = asof + timedelta(days=lookahead_days)
    try:
        events = await provider.get_earnings_calendar(asof, end)
    except Exception:
        return 0

    rows_written = 0
    for event in events:
        symbol = event.get("symbol", "")
        announce_date = event.get("announce_date")
        if not symbol or not announce_date:
            continue
        if db is not None:
            await db.execute(
                """
                INSERT INTO earnings_calendar (symbol, announce_date, confirmed, ingested_at)
                VALUES ($1, $2, $3, NOW())
                ON CONFLICT (symbol, announce_date) DO UPDATE
                SET confirmed=$3, ingested_at=NOW()
                """,
                symbol, announce_date, event.get("confirmed", False),
            )
        rows_written += 1
    return rows_written
