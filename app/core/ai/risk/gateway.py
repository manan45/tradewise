"""RiskGateway — single chokepoint between scenarios and the broker."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

from app.core.ports.broker import Order, OrderSide, OrderType, Position
from .blackouts import EventBlackout
from .circuit_breaker import CircuitBreaker
from .sizing import SizingInputs, kelly_fraction, position_size_shares


class RiskRejection(str, Enum):
    BREAKER_TRIPPED = "breaker_tripped"
    BLACKOUT = "blackout"
    EXPOSURE_CAP = "exposure_cap"
    CORRELATION_CAP = "correlation_cap"
    SIZING_ZERO = "sizing_zero"
    KELLY_NEGATIVE = "kelly_negative"


@dataclass(frozen=True)
class RiskDecision:
    approved: bool
    order: Order | None
    rejection: RiskRejection | None = None
    notes: dict[str, Any] | None = None


class RiskGateway:
    def __init__(
        self,
        breaker: CircuitBreaker,
        blackouts: EventBlackout,
        max_per_asset_pct: float = 0.10,
        max_per_sector_pct: float = 0.30,
        max_correlation: float = 0.7,
    ):
        self.breaker = breaker
        self.blackouts = blackouts
        self.max_per_asset_pct = max_per_asset_pct
        self.max_per_sector_pct = max_per_sector_pct
        self.max_correlation = max_correlation

    def evaluate(
        self,
        intent: dict[str, Any],         # plan from a ScenarioSignal
        sizing: SizingInputs,
        ts: datetime,
        account_equity: float,
        open_positions: list[Position],
        sector_lookup: dict[str, str],
        correlation_matrix: dict[tuple[str, str], float] | None = None,
    ) -> RiskDecision:
        symbol = intent["symbol"]
        side = intent.get("side", "buy")
        entry_price = intent.get("entry", 0.0)

        # 1. Circuit breaker
        if self.breaker.tripped:
            return RiskDecision(approved=False, order=None,
                                rejection=RiskRejection.BREAKER_TRIPPED)

        # 2. Event blackout
        blocked = self.blackouts.is_blocked(symbol, ts)
        if blocked is not None:
            return RiskDecision(approved=False, order=None,
                                rejection=RiskRejection.BLACKOUT,
                                notes={"reason": blocked.reason})

        # 3. Per-asset exposure cap
        existing_market_value = sum(
            abs(p.market_value) for p in open_positions if p.symbol == symbol
        )
        if account_equity > 0 and (existing_market_value / account_equity) >= self.max_per_asset_pct:
            return RiskDecision(approved=False, order=None,
                                rejection=RiskRejection.EXPOSURE_CAP)

        # 4. Correlation cap
        if correlation_matrix is not None:
            for pos in open_positions:
                corr = correlation_matrix.get((symbol, pos.symbol)) or \
                       correlation_matrix.get((pos.symbol, symbol))
                if corr is not None and abs(corr) > self.max_correlation:
                    return RiskDecision(approved=False, order=None,
                                        rejection=RiskRejection.CORRELATION_CAP,
                                        notes={"correlated_with": pos.symbol, "corr": corr})

        # 5. Kelly sizing
        raw_kelly = kelly_fraction(sizing.p_win, sizing.win_loss_ratio)
        if raw_kelly <= 0:
            return RiskDecision(approved=False, order=None,
                                rejection=RiskRejection.KELLY_NEGATIVE)

        shares = position_size_shares(sizing, account_equity, entry_price)
        if shares <= 0:
            return RiskDecision(approved=False, order=None,
                                rejection=RiskRejection.SIZING_ZERO)

        import uuid
        order = Order(
            client_order_id=str(uuid.uuid4()),
            symbol=symbol,
            side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
            qty=float(shares),
            order_type=OrderType.LIMIT,
            limit_price=entry_price,
        )
        return RiskDecision(approved=True, order=order,
                            notes={"shares": shares, "kelly_raw": raw_kelly})
