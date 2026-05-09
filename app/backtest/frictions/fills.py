"""Fill model — when does each order type fill, at what price.

Conservative defaults:
- MARKET: next bar open (never current bar close).
- LIMIT: same bar if low <= limit (buy) or high >= limit (sell), else next bar
  if it touches; assume worst-case fill at limit.
- STOP: same bar if stop is touched, fill at stop +/- slippage.
"""
from __future__ import annotations

from datetime import timezone

from app.core.ports.broker import Fill, Order, OrderSide, OrderType
from app.core.ports.market_data import Bar


class FillModel:
    def __init__(self):
        pass

    def attempt_fill(self, order: Order, bar: Bar,
                     prior_bar: Bar | None) -> Fill | None:
        if order.order_type == OrderType.MARKET:
            fill_price = bar.open
            return Fill(
                order_id=order.client_order_id,
                symbol=order.symbol,
                side=order.side,
                qty=order.qty,
                price=fill_price,
                ts=bar.ts,
            )

        elif order.order_type == OrderType.LIMIT:
            limit = order.limit_price
            if limit is None:
                return None
            if order.side == OrderSide.BUY and bar.low <= limit:
                return Fill(
                    order_id=order.client_order_id,
                    symbol=order.symbol,
                    side=order.side,
                    qty=order.qty,
                    price=limit,
                    ts=bar.ts,
                )
            elif order.side == OrderSide.SELL and bar.high >= limit:
                return Fill(
                    order_id=order.client_order_id,
                    symbol=order.symbol,
                    side=order.side,
                    qty=order.qty,
                    price=limit,
                    ts=bar.ts,
                )

        elif order.order_type == OrderType.STOP:
            stop = order.stop_price
            if stop is None:
                return None
            if order.side == OrderSide.SELL and bar.low <= stop:
                return Fill(
                    order_id=order.client_order_id,
                    symbol=order.symbol,
                    side=order.side,
                    qty=order.qty,
                    price=stop,
                    ts=bar.ts,
                )
            elif order.side == OrderSide.BUY and bar.high >= stop:
                return Fill(
                    order_id=order.client_order_id,
                    symbol=order.symbol,
                    side=order.side,
                    qty=order.qty,
                    price=stop,
                    ts=bar.ts,
                )

        return None
