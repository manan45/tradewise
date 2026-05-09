"""CFTC Commitment of Traders weekly ingest (futures positioning)."""
from __future__ import annotations

from datetime import date

from app.core.ports.macro import MacroDataProvider


COT_MARKETS = ("GOLD", "SILVER", "COPPER", "WTI_CRUDE", "CORN", "WHEAT",
               "SOYBEANS", "NATURAL_GAS", "SP500_EMINI")


async def run(asof: date, provider: MacroDataProvider,
              markets: tuple[str, ...] = COT_MARKETS,
              db=None) -> int:
    rows_written = 0
    for market in markets:
        try:
            report = await provider.get_cot_report(market, asof)
        except Exception:
            continue
        if not report:
            continue
        if db is not None:
            await db.execute(
                """
                INSERT INTO cot_weekly (as_of, symbol, comm_long, comm_short, noncomm_long, noncomm_short, ingested_at)
                VALUES ($1, $2, $3, $4, $5, $6, NOW())
                ON CONFLICT (as_of, symbol) DO UPDATE
                SET comm_long=$3, comm_short=$4, noncomm_long=$5, noncomm_short=$6, ingested_at=NOW()
                """,
                asof, market,
                report.get("comm_long", 0), report.get("comm_short", 0),
                report.get("noncomm_long", 0), report.get("noncomm_short", 0),
            )
        rows_written += 1
    return rows_written
