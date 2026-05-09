"""Broker port — order placement, fills, positions, account.

Implementations: AlpacaBroker (US equities, paper + live), IBKRBroker (futures),
SimBroker (backtest, fills computed by frictions module).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Protocol


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


@dataclass(frozen=True)
class Order:
    client_order_id: str
    symbol: str
    side: OrderSide
    qty: float
    order_type: OrderType
    limit_price: float | None = None
    stop_price: float | None = None
    tif: str = "day"
    extended_hours: bool = False


@dataclass(frozen=True)
class Fill:
    order_id: str
    symbol: str
    side: OrderSide
    qty: float
    price: float
    ts: datetime
    commission: float = 0.0


@dataclass(frozen=True)
class Position:
    symbol: str
    qty: float
    avg_price: float
    market_value: float
    unrealized_pnl: float


class BrokerAdapter(Protocol):
    async def submit(self, order: Order) -> str:
        """Returns broker-side order id."""
        ...

    async def cancel(self, broker_order_id: str) -> None:
        ...

    async def get_position(self, symbol: str) -> Position | None:
        ...

    async def list_positions(self) -> list[Position]:
        ...

    async def get_account_equity(self) -> float:
        ...
