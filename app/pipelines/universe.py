"""Universe membership refresh — S&P 500, Russell 2000, commodity universe."""
from __future__ import annotations

from datetime import date


UNIVERSES = ("sp500", "russell2000", "commodities_core")

_COMMODITIES_CORE = [
    "GC", "SI", "HG", "CL", "NG", "ZC", "ZW", "ZS", "ES",
]


async def run(asof: date, universes: tuple[str, ...] = UNIVERSES,
              db=None) -> int:
    rows_written = 0
    for universe in universes:
        symbols = _get_symbols(universe)
        for symbol in symbols:
            if db is not None:
                await db.execute(
                    """
                    INSERT INTO universe_membership (universe, symbol, entered_at)
                    VALUES ($1, $2, $3)
                    ON CONFLICT (universe, symbol) DO NOTHING
                    """,
                    universe, symbol, asof,
                )
            rows_written += 1
    return rows_written


def _get_symbols(universe: str) -> list[str]:
    if universe == "commodities_core":
        return _COMMODITIES_CORE
    # For sp500 and russell2000 a real impl would pull from Wikipedia or a vendor.
    # Returning empty list until a provider is wired.
    return []
