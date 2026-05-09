"""Event blackout rules.

Per spec: no new opens within (T-1d, T+1d) of earnings; no new opens within
30 min of scheduled macro releases (FOMC, NFP, CPI). Existing positions can
still be exited.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable


@dataclass(frozen=True)
class BlackoutWindow:
    symbol: str | None       # None means market-wide
    start: datetime
    end: datetime
    reason: str


class EventBlackout:
    def __init__(self, windows: Iterable[BlackoutWindow]):
        self._windows: list[BlackoutWindow] = sorted(windows, key=lambda w: w.start)

    def is_blocked(self, symbol: str, ts: datetime) -> BlackoutWindow | None:
        for w in self._windows:
            if w.start > ts:
                break
            if w.end < ts:
                continue
            if w.symbol is None or w.symbol == symbol:
                return w
        return None

    @classmethod
    def from_earnings_calendar(
        cls,
        rows: list[dict],
        before: timedelta = timedelta(days=1),
        after: timedelta = timedelta(days=1),
    ) -> "EventBlackout":
        windows: list[BlackoutWindow] = []
        for row in rows:
            announce = row["announce_date"]
            sym = row.get("symbol")
            windows.append(BlackoutWindow(
                symbol=sym,
                start=announce - before,
                end=announce + after,
                reason="earnings",
            ))
        return cls(windows)
