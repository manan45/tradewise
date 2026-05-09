"""IBKR adapter for futures (gold, copper, oil, grains, currencies)."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import AsyncIterator, Iterable

from app.core.ports.broker import Order, OrderSide, Position
from app.core.ports.market_data import Bar, Quote

_FUTURES_SYMBOL_MAP = {
    "GC": ("COMEX", "Gold"),
    "SI": ("COMEX", "Silver"),
    "HG": ("COMEX", "Copper"),
    "CL": ("NYMEX", "Crude Oil"),
    "NG": ("NYMEX", "Natural Gas"),
    "ZC": ("CBOT", "Corn"),
    "ZW": ("CBOT", "Wheat"),
    "ZS": ("CBOT", "Soybeans"),
    "ES": ("CME", "E-mini S&P 500"),
}

_TF_MAP = {
    "1m": "1 min",
    "5m": "5 mins",
    "1h": "1 hour",
    "1d": "1 day",
}


class IBKRMarketData:
    def __init__(self, host: str = "127.0.0.1", port: int = 7497,
                 client_id: int = 1):
        self.host = host
        self.port = port
        self.client_id = client_id
        self._ib = None

    def _get_ib(self):
        if self._ib is None:
            from ib_insync import IB
            ib = IB()
            ib.connect(self.host, self.port, clientId=self.client_id)
            self._ib = ib
        return self._ib

    def _make_contract(self, symbol: str):
        from ib_insync import Future
        exchange, _ = _FUTURES_SYMBOL_MAP.get(symbol, ("SMART", symbol))
        return Future(symbol=symbol, exchange=exchange, currency="USD")

    async def get_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        interval: str = "1d",
    ) -> list[Bar]:
        import asyncio
        ib = self._get_ib()
        contract = self._make_contract(symbol)
        duration_days = max(1, (end - start).days + 1)
        duration_str = f"{duration_days} D"
        bar_size = _TF_MAP.get(interval, "1 day")
        loop = asyncio.get_event_loop()
        raw_bars = await loop.run_in_executor(
            None,
            lambda: ib.reqHistoricalData(
                contract,
                endDateTime=end.strftime("%Y%m%d %H:%M:%S"),
                durationStr=duration_str,
                barSizeSetting=bar_size,
                whatToShow="TRADES",
                useRTH=True,
                formatDate=2,
            ),
        )
        result = []
        for b in raw_bars:
            ts = b.date
            if not hasattr(ts, "tzinfo") or ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            result.append(Bar(
                symbol=symbol,
                ts=ts,
                open=b.open,
                high=b.high,
                low=b.low,
                close=b.close,
                volume=b.volume,
                vwap=b.average,
                interval=interval,
            ))
        return result

    async def get_latest_quote(self, symbol: str) -> Quote:
        import asyncio
        ib = self._get_ib()
        contract = self._make_contract(symbol)
        loop = asyncio.get_event_loop()
        ticker = await loop.run_in_executor(
            None,
            lambda: ib.reqMktData(contract, "", False, False),
        )
        ib.sleep(1)
        ts = datetime.now(tz=timezone.utc)
        return Quote(
            symbol=symbol,
            ts=ts,
            bid=ticker.bid or 0.0,
            ask=ticker.ask or 0.0,
            bid_size=ticker.bidSize or 0,
            ask_size=ticker.askSize or 0,
        )

    async def subscribe_bars(
        self,
        symbols: Iterable[str],
        interval: str = "1m",
    ) -> AsyncIterator[Bar]:
        import asyncio
        ib = self._get_ib()
        queue: asyncio.Queue[Bar] = asyncio.Queue()
        syms = list(symbols)
        contracts = [self._make_contract(s) for s in syms]

        def on_bar(bars, has_new_bar):
            if not has_new_bar:
                return
            for b, sym in zip(bars, syms):
                ts = datetime.now(tz=timezone.utc)
                queue.put_nowait(Bar(
                    symbol=sym,
                    ts=ts,
                    open=b.open,
                    high=b.high,
                    low=b.low,
                    close=b.close,
                    volume=b.volume,
                    interval=interval,
                ))

        for contract in contracts:
            ib.reqRealTimeBars(contract, 5, "TRADES", False, callback=on_bar)

        while True:
            yield await queue.get()

    async def aclose(self) -> None:
        if self._ib is not None:
            self._ib.disconnect()


class IBKRBroker:
    def __init__(self, host: str = "127.0.0.1", port: int = 7497,
                 client_id: int = 2):
        self.host = host
        self.port = port
        self.client_id = client_id
        self._ib = None

    def _get_ib(self):
        if self._ib is None:
            from ib_insync import IB
            ib = IB()
            ib.connect(self.host, self.port, clientId=self.client_id)
            self._ib = ib
        return self._ib

    def _make_contract(self, symbol: str):
        from ib_insync import Future
        exchange, _ = _FUTURES_SYMBOL_MAP.get(symbol, ("SMART", symbol))
        return Future(symbol=symbol, exchange=exchange, currency="USD")

    async def submit(self, order: Order) -> str:
        import asyncio
        from ib_insync import MarketOrder, LimitOrder
        ib = self._get_ib()
        contract = self._make_contract(order.symbol)
        action = "BUY" if order.side == OrderSide.BUY else "SELL"
        if order.limit_price:
            ib_order = LimitOrder(action, int(order.qty), order.limit_price)
        else:
            ib_order = MarketOrder(action, int(order.qty))
        loop = asyncio.get_event_loop()
        trade = await loop.run_in_executor(
            None,
            lambda: ib.placeOrder(contract, ib_order),
        )
        return str(trade.order.orderId)

    async def cancel(self, broker_order_id: str) -> None:
        import asyncio
        ib = self._get_ib()
        order_id = int(broker_order_id)
        for trade in ib.trades():
            if trade.order.orderId == order_id:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, lambda: ib.cancelOrder(trade.order))
                return

    async def get_position(self, symbol: str) -> Position | None:
        import asyncio
        ib = self._get_ib()
        loop = asyncio.get_event_loop()
        positions = await loop.run_in_executor(None, ib.positions)
        for p in positions:
            if p.contract.symbol == symbol:
                mv = p.position * p.avgCost
                return Position(
                    symbol=symbol,
                    qty=p.position,
                    avg_price=p.avgCost,
                    market_value=mv,
                    unrealized_pnl=0.0,
                )
        return None

    async def list_positions(self) -> list[Position]:
        import asyncio
        ib = self._get_ib()
        loop = asyncio.get_event_loop()
        positions = await loop.run_in_executor(None, ib.positions)
        return [
            Position(
                symbol=p.contract.symbol,
                qty=p.position,
                avg_price=p.avgCost,
                market_value=p.position * p.avgCost,
                unrealized_pnl=0.0,
            )
            for p in positions
        ]

    async def get_account_equity(self) -> float:
        import asyncio
        ib = self._get_ib()
        loop = asyncio.get_event_loop()
        summary = await loop.run_in_executor(None, ib.accountSummary)
        for item in summary:
            if item.tag == "NetLiquidation":
                return float(item.value)
        return 0.0

    async def aclose(self) -> None:
        if self._ib is not None:
            self._ib.disconnect()
