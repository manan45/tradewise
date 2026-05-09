"""Commission models — per-broker, per-asset."""
from __future__ import annotations

from app.core.ports.broker import Order


class CommissionModel:
    def __init__(
        self,
        equity_per_share: float = 0.005,
        equity_min: float = 1.0,
        future_per_contract: float = 2.25,
    ):
        self.equity_per_share = equity_per_share
        self.equity_min = equity_min
        self.future_per_contract = future_per_contract

    def cost(self, order: Order, fill_price: float,
             asset_class: str = "equity") -> float:
        if asset_class == "future":
            return self.future_per_contract * order.qty
        return max(self.equity_min, self.equity_per_share * order.qty)
