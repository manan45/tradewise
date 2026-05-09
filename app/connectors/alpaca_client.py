"""Alpaca adapter — implements MarketDataProvider AND BrokerAdapter."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import AsyncIterator, Iterable

import httpx

from app.core.ports.broker import BrokerAdapter, Fill, Order, OrderSide, OrderType, Position
from app.core.ports.market_data import Bar, Quote


class BrokerRejected(Exception):
    pass


class AlpacaMarketData:
    def __init__(self, key_id: str, secret: str, paper: bool = True):
        self.key_id = key_id
        self.secret = secret
        self.paper = paper
        self._base = "https://data.alpaca.markets"
        self._client: httpx.AsyncClient | None = None

    def _headers(self) -> dict[str, str]:
        return {"APCA-API-KEY-ID": self.key_id, "APCA-API-SECRET-KEY": self.secret}

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=10.0)
        return self._client

    async def aclose(self) -> None:
        if self._client:
            await self._client.aclose()

    async def get_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        interval: str = "1d",
    ) -> list[Bar]:
        tf_map = {"1m": "1Min", "5m": "5Min", "1h": "1Hour", "1d": "1Day"}
        timeframe = tf_map.get(interval, "1Day")
        client = self._get_client()
        url = f"{self._base}/v2/stocks/{symbol}/bars"
        params = {
            "timeframe": timeframe,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "limit": 10000,
            "adjustment": "split",
        }
        bars: list[Bar] = []
        while True:
            resp = await client.get(url, headers=self._headers(), params=params)
            resp.raise_for_status()
            data = resp.json()
            for r in data.get("bars", []):
                ts = datetime.fromisoformat(r["t"].replace("Z", "+00:00"))
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
            next_token = data.get("next_page_token")
            if not next_token:
                break
            params["page_token"] = next_token
        return bars

    async def get_latest_quote(self, symbol: str) -> Quote:
        client = self._get_client()
        resp = await client.get(
            f"{self._base}/v2/stocks/{symbol}/quotes/latest",
            headers=self._headers(),
        )
        resp.raise_for_status()
        q = resp.json()["quote"]
        ts = datetime.fromisoformat(q["t"].replace("Z", "+00:00"))
        return Quote(symbol=symbol, ts=ts, bid=q["bp"], ask=q["ap"],
                     bid_size=q.get("bs", 0), ask_size=q.get("as", 0))

    async def subscribe_bars(
        self,
        symbols: Iterable[str],
        interval: str = "1m",
    ) -> AsyncIterator[Bar]:
        from alpaca.data.live import StockDataStream
        import asyncio
        syms = list(symbols)
        stream = StockDataStream(self.key_id, self.secret)
        queue: asyncio.Queue[Bar] = asyncio.Queue()

        async def handler(bar):
            ts = bar.timestamp
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            await queue.put(Bar(
                symbol=bar.symbol,
                ts=ts,
                open=bar.open,
                high=bar.high,
                low=bar.low,
                close=bar.close,
                volume=bar.volume,
                vwap=bar.vwap,
                interval=interval,
            ))

        stream.subscribe_bars(handler, *syms)
        asyncio.create_task(stream.run())
        while True:
            yield await queue.get()


class AlpacaBroker:
    def __init__(self, key_id: str, secret: str, paper: bool = True):
        self.key_id = key_id
        self.secret = secret
        self.paper = paper
        base = "https://paper-api.alpaca.markets" if paper else "https://api.alpaca.markets"
        self._base = base
        self._client: httpx.AsyncClient | None = None

    def _headers(self) -> dict[str, str]:
        return {"APCA-API-KEY-ID": self.key_id, "APCA-API-SECRET-KEY": self.secret}

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=10.0)
        return self._client

    async def aclose(self) -> None:
        if self._client:
            await self._client.aclose()

    async def submit(self, order: Order) -> str:
        side_map = {OrderSide.BUY: "buy", OrderSide.SELL: "sell"}
        type_map = {OrderType.MARKET: "market", OrderType.LIMIT: "limit",
                    OrderType.STOP: "stop", OrderType.STOP_LIMIT: "stop_limit"}
        payload: dict = {
            "symbol": order.symbol,
            "qty": str(int(order.qty)),
            "side": side_map[order.side],
            "type": type_map[order.order_type],
            "time_in_force": order.tif,
            "client_order_id": order.client_order_id,
        }
        if order.limit_price is not None:
            payload["limit_price"] = str(order.limit_price)
        if order.stop_price is not None:
            payload["stop_price"] = str(order.stop_price)
        client = self._get_client()
        resp = await client.post(f"{self._base}/v2/orders", headers=self._headers(), json=payload)
        if resp.status_code == 422:
            raise BrokerRejected(resp.text)
        resp.raise_for_status()
        return resp.json()["id"]

    async def cancel(self, broker_order_id: str) -> None:
        client = self._get_client()
        resp = await client.delete(f"{self._base}/v2/orders/{broker_order_id}",
                                   headers=self._headers())
        if resp.status_code == 404:
            return
        resp.raise_for_status()

    async def get_position(self, symbol: str) -> Position | None:
        client = self._get_client()
        resp = await client.get(f"{self._base}/v2/positions/{symbol}",
                                headers=self._headers())
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        p = resp.json()
        return Position(
            symbol=p["symbol"],
            qty=float(p["qty"]),
            avg_price=float(p["avg_entry_price"]),
            market_value=float(p["market_value"]),
            unrealized_pnl=float(p["unrealized_pl"]),
        )

    async def list_positions(self) -> list[Position]:
        client = self._get_client()
        resp = await client.get(f"{self._base}/v2/positions", headers=self._headers())
        resp.raise_for_status()
        return [
            Position(
                symbol=p["symbol"],
                qty=float(p["qty"]),
                avg_price=float(p["avg_entry_price"]),
                market_value=float(p["market_value"]),
                unrealized_pnl=float(p["unrealized_pl"]),
            )
            for p in resp.json()
        ]

    async def get_account_equity(self) -> float:
        client = self._get_client()
        resp = await client.get(f"{self._base}/v2/account", headers=self._headers())
        resp.raise_for_status()
        return float(resp.json()["equity"])
