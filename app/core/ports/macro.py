"""Macro data port — FRED series, COT reports, VIX term structure, DXY.

Implementations: FredClient, CftcCotClient, CboeVixClient, plus a fixture
provider for tests.
"""
from __future__ import annotations

from datetime import date
from typing import Protocol


class MacroDataProvider(Protocol):
    async def get_series(
        self,
        series_id: str,
        start: date,
        end: date,
    ) -> list[tuple[date, float]]:
        ...

    async def get_cot_report(
        self,
        market: str,
        as_of: date,
    ) -> dict[str, float]:
        """Commercial / non-commercial / non-reportable net positions."""
        ...

    async def get_vix_term_structure(self, as_of: date) -> dict[str, float]:
        """{'VIX9D': ..., 'VIX': ..., 'VIX3M': ..., 'VIX6M': ...}"""
        ...
