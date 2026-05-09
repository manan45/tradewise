"""Market data port — historical + streaming bars/quotes for equities & futures.

Implementations: PolygonClient (US equities), AlpacaClient (intraday + paper),
IBKRClient (futures), YahooFallback (smoke), and an in-memory replayer for
backtest mode.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import AsyncIterator, Iterable, Protocol


@dataclass(frozen=True)
class Bar:
    symbol: str
    ts: datetime         # bar close time, tz-aware (UTC)
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: float | None = None
    interval: str = "1d"  # "1m" | "5m" | "1h" | "1d"


@dataclass(frozen=True)
class Quote:
    symbol: str
    ts: datetime
    bid: float
    ask: float
    bid_size: float
    ask_size: float


class MarketDataProvider(Protocol):
    """Pull-mode interface; streaming is opt-in via subscribe()."""

    async def get_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        interval: str = "1d",
    ) -> list[Bar]:
        ...

    async def get_latest_quote(self, symbol: str) -> Quote:
        ...

    async def subscribe_bars(
        self,
        symbols: Iterable[str],
        interval: str = "1m",
    ) -> AsyncIterator[Bar]:
        ...
