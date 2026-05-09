"""News + filings port — pulls headlines and SEC filings for sentiment + RAG."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol


@dataclass(frozen=True)
class NewsItem:
    id: str
    symbol: str | None
    ts: datetime
    headline: str
    body: str
    source: str
    url: str | None = None


class NewsProvider(Protocol):
    async def fetch_recent(
        self,
        symbol: str | None,
        since: datetime,
        limit: int = 100,
    ) -> list[NewsItem]:
        ...
