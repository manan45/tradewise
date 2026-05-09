"""Slippage models.

Equity: cost = k * (order_qty / bar_volume) ** 0.5 — square-root
participation impact, k calibrated per liquidity bucket.
Futures: cost = ticks * tick_size, where ticks depends on contract month.
"""
from __future__ import annotations

import math
from typing import Protocol

from app.core.ports.broker import Order


class SlippageModel(Protocol):
    def cost_per_share(self, order: Order, bar_volume: float,
                       last_price: float) -> float:
        ...


class EquitySlippage:
    def __init__(self, k_by_liquidity: dict[str, float] | None = None):
        self.k_by_liquidity = k_by_liquidity or {"high": 0.0001, "mid": 0.0005,
                                                  "low": 0.002}

    def cost_per_share(self, order: Order, bar_volume: float,
                       last_price: float) -> float:
        if order.qty <= 0:
            return 0.0
        k = self.k_by_liquidity.get("mid", 0.0005)
        participation = order.qty / max(bar_volume, 1.0)
        return k * last_price * math.sqrt(participation)


class FuturesSlippage:
    def __init__(self, tick_size: dict[str, float],
                 cushion_ticks: dict[str, int]):
        self.tick_size = tick_size
        self.cushion_ticks = cushion_ticks

    def cost_per_share(self, order: Order, bar_volume: float,
                       last_price: float) -> float:
        ts = self.tick_size.get(order.symbol, 0.25)
        ticks = self.cushion_ticks.get(order.symbol, 1)
        participation_ticks = math.ceil(order.qty / max(bar_volume, 1.0) * 10)
        return ts * max(ticks, participation_ticks)
