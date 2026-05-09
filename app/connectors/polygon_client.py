"""Polygon.io adapter implementing MarketDataProvider for US equities."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import AsyncIterator, Iterable

import httpx

from app.core.ports.market_data import Bar, MarketDataProvider, Quote

_TIMEFRAME_MAP = {
    "1m": ("minute", 1),
    "5m": ("minute", 5),
    "1h": ("hour", 1),
    "1d": ("day", 1),
}


class PolygonClient:
    def __init__(self, api_key: str, base_url: str = "https://api.polygon.io"):
        self.api_key = api_key
        self.base_url = base_url
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=10.0)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def get_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        interval: str = "1d",
    ) -> list[Bar]:
        timespan, multiplier = _TIMEFRAME_MAP.get(interval, ("day", 1))
        from_str = start.strftime("%Y-%m-%d")
        to_str = end.strftime("%Y-%m-%d")
        url = f"{self.base_url}/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{from_str}/{to_str}"
        params = {"apiKey": self.api_key, "adjusted": "true", "sort": "asc", "limit": 50000}
        bars: list[Bar] = []
        client = self._get_client()
        while url:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
            for r in data.get("results", []):
                ts = datetime.fromtimestamp(r["t"] / 1000, tz=timezone.utc)
                bars.append(Bar(
                    symbol=symbol,
                    ts=ts,
                    open=r["o"],
                    high=r["h"],
                    low=r["l"],
                    close=r["c"],
                    volume=r.get("v", 0),
                    vwap=r.get("vw"),
                    interval=interval,
                ))
            url = data.get("next_url")
            params = {"apiKey": self.api_key}
        # dedupe by ts
        seen: set[datetime] = set()
        deduped: list[Bar] = []
        for b in bars:
            if b.ts not in seen:
                seen.add(b.ts)
                deduped.append(b)
        return deduped

    async def get_latest_quote(self, symbol: str) -> Quote:
        client = self._get_client()
        url = f"{self.base_url}/v2/last/nbbo/{symbol}"
        resp = await client.get(url, params={"apiKey": self.api_key})
        resp.raise_for_status()
        r = resp.json()["results"]
        ts = datetime.fromtimestamp(r["t"] / 1_000_000_000, tz=timezone.utc)
        return Quote(
            symbol=symbol,
            ts=ts,
            bid=r["P"],
            ask=r["p"],
            bid_size=r.get("S", 0),
            ask_size=r.get("s", 0),
        )

    async def subscribe_bars(
        self,
        symbols: Iterable[str],
        interval: str = "1m",
    ) -> AsyncIterator[Bar]:
        import websockets
        import json
        syms = list(symbols)
        backoff = 1.0
        ws_url = "wss://socket.polygon.io/stocks"
        channel = "AM" if interval == "1m" else "A"
        while True:
            try:
                async with websockets.connect(ws_url) as ws:
                    await ws.send(json.dumps({"action": "auth", "params": self.api_key}))
                    auth_msg = json.loads(await ws.recv())
                    subs = ",".join(f"{channel}.{s}" for s in syms)
                    await ws.send(json.dumps({"action": "subscribe", "params": subs}))
                    backoff = 1.0
                    async for raw in ws:
                        messages = json.loads(raw)
                        for m in messages:
                            if m.get("ev") not in ("AM", "A"):
                                continue
                            ts = datetime.fromtimestamp(m["e"] / 1000, tz=timezone.utc)
                            yield Bar(
                                symbol=m["sym"],
                                ts=ts,
                                open=m["o"],
                                high=m["h"],
                                low=m["l"],
                                close=m["c"],
                                volume=m.get("av", 0),
                                vwap=m.get("vw"),
                                interval=interval,
                            )
            except Exception:
                await asyncio.sleep(min(backoff, 60.0))
                backoff *= 2
