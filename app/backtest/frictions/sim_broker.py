"""SimBroker — in-memory BrokerAdapter that backtest mode wires in.

Holds a virtual portfolio and resolves orders via FillModel + slippage +
commissions on each replayed bar.
"""
from __future__ import annotations

import uuid

from app.core.ports.broker import BrokerAdapter, Fill, Order, OrderSide, Position
from app.core.ports.market_data import Bar
from .commissions import CommissionModel
from .fills import FillModel
from .slippage import SlippageModel


class SimBroker:
    def __init__(
        self,
        starting_equity: float,
        slippage: SlippageModel,
        commissions: CommissionModel,
        fills: FillModel,
    ):
        self.equity = starting_equity
        self.slippage = slippage
        self.commissions = commissions
        self.fills = fills
        self._positions: dict[str, Position] = {}
        self._working_orders: list[Order] = []
        self._id_seq: int = 0
        self.fill_log: list[Fill] = []

    async def submit(self, order: Order) -> str:
        self._working_orders.append(order)
        return order.client_order_id

    async def cancel(self, broker_order_id: str) -> None:
        self._working_orders = [
            o for o in self._working_orders if o.client_order_id != broker_order_id
        ]

    async def get_position(self, symbol: str) -> Position | None:
        return self._positions.get(symbol)

    async def list_positions(self) -> list[Position]:
        return list(self._positions.values())

    async def get_account_equity(self) -> float:
        return self.equity

    def process_bar(self, bar: Bar, prior_bar: Bar | None = None,
                    asset_class: str = "equity") -> list[Fill]:
        """Attempt to fill all working orders against the current bar."""
        filled: list[Fill] = []
        remaining: list[Order] = []
        for order in self._working_orders:
            if order.symbol != bar.symbol:
                remaining.append(order)
                continue
            fill = self.fills.attempt_fill(order, bar, prior_bar)
            if fill is None:
                remaining.append(order)
                continue
            slip = self.slippage.cost_per_share(order, bar.volume, bar.close)
            comm = self.commissions.cost(order, fill.price, asset_class)
            actual_price = (fill.price + slip) if order.side == OrderSide.BUY else (fill.price - slip)
            fill = Fill(
                order_id=fill.order_id,
                symbol=fill.symbol,
                side=fill.side,
                qty=fill.qty,
                price=actual_price,
                ts=fill.ts,
                commission=comm,
            )
            self._apply_fill(fill)
            filled.append(fill)
        self._working_orders = remaining
        self.fill_log.extend(filled)
        return filled

    def mark_to_market(self, prices: dict[str, float]) -> None:
        """Update position market values and equity using current prices."""
        for symbol, pos in list(self._positions.items()):
            price = prices.get(symbol, pos.avg_price)
            mv = pos.qty * price
            upnl = (price - pos.avg_price) * pos.qty
            self._positions[symbol] = Position(
                symbol=symbol,
                qty=pos.qty,
                avg_price=pos.avg_price,
                market_value=mv,
                unrealized_pnl=upnl,
            )

    def _apply_fill(self, fill: Fill) -> None:
        pos = self._positions.get(fill.symbol)
        if fill.side == OrderSide.BUY:
            cost = fill.qty * fill.price + fill.commission
            self.equity -= cost
            if pos is None:
                self._positions[fill.symbol] = Position(
                    symbol=fill.symbol, qty=fill.qty,
                    avg_price=fill.price, market_value=fill.qty * fill.price,
                    unrealized_pnl=0.0,
                )
            else:
                new_qty = pos.qty + fill.qty
                new_avg = (pos.avg_price * pos.qty + fill.price * fill.qty) / new_qty
                self._positions[fill.symbol] = Position(
                    symbol=fill.symbol, qty=new_qty,
                    avg_price=new_avg, market_value=new_qty * fill.price,
                    unrealized_pnl=0.0,
                )
        else:  # SELL
            proceeds = fill.qty * fill.price - fill.commission
            self.equity += proceeds
            if pos is not None:
                new_qty = pos.qty - fill.qty
                if new_qty <= 0:
                    del self._positions[fill.symbol]
                else:
                    self._positions[fill.symbol] = Position(
                        symbol=fill.symbol, qty=new_qty,
                        avg_price=pos.avg_price,
                        market_value=new_qty * fill.price,
                        unrealized_pnl=(fill.price - pos.avg_price) * new_qty,
                    )
