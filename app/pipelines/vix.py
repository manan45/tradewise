"""VIX term structure (VIX9D, VIX, VIX3M, VIX6M) — daily snapshot."""
from __future__ import annotations

from datetime import date

from app.core.ports.macro import MacroDataProvider


async def run(asof: date, provider: MacroDataProvider, db=None) -> int:
    term = await provider.get_vix_term_structure(asof)
    if db is not None:
        await db.execute(
            """
            INSERT INTO vix_term (asof, vix, vix9d, vix3m, vix6m, vix_v3m_ratio, ingested_at)
            VALUES ($1, $2, $3, $4, $5, $6, NOW())
            ON CONFLICT (asof) DO UPDATE
            SET vix=$2, vix9d=$3, vix3m=$4, vix6m=$5, vix_v3m_ratio=$6, ingested_at=NOW()
            """,
            asof,
            term.get("VIX"),
            term.get("VIX9D"),
            term.get("VIX3M"),
            term.get("VIX6M"),
            term.get("VIX_V3M_RATIO"),
        )
    return 1
