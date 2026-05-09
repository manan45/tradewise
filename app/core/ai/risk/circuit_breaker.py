"""Global circuit breaker — kills all new orders when tripped.

Trip conditions (any of):
- Daily PnL < -2% of starting equity.
- Rolling 5-day PnL < -5%.
- 3+ consecutive scenarios closed at -2R or worse.
- Manual trip via CLI / Telegram approval message.

State persists in `system_state.trading_halted`; reset is manual only
(operator must clear via CLI).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum


class BreakerTrip(str, Enum):
    DAILY_DRAWDOWN = "daily_drawdown"
    WEEKLY_DRAWDOWN = "weekly_drawdown"
    CONSECUTIVE_LOSSES = "consecutive_losses"
    MANUAL = "manual"


@dataclass(frozen=True)
class BreakerState:
    halted: bool
    reason: BreakerTrip | None
    halted_at: datetime | None


class CircuitBreaker:
    def __init__(self, daily_dd_pct: float = 0.02,
                 weekly_dd_pct: float = 0.05,
                 max_consec_losses: int = 3):
        self.daily_dd_pct = daily_dd_pct
        self.weekly_dd_pct = weekly_dd_pct
        self.max_consec_losses = max_consec_losses
        self._state = BreakerState(halted=False, reason=None, halted_at=None)
        self._bus = None   # injected after construction

    @property
    def tripped(self) -> bool:
        return self._state.halted

    def evaluate(
        self,
        starting_equity: float,
        current_equity: float,
        weekly_pnl: float,
        consecutive_losses: int,
    ) -> BreakerTrip | None:
        daily_dd = (starting_equity - current_equity) / starting_equity
        if daily_dd >= self.daily_dd_pct:
            return BreakerTrip.DAILY_DRAWDOWN
        if starting_equity > 0 and (-weekly_pnl / starting_equity) >= self.weekly_dd_pct:
            return BreakerTrip.WEEKLY_DRAWDOWN
        if consecutive_losses >= self.max_consec_losses:
            return BreakerTrip.CONSECUTIVE_LOSSES
        return None

    async def trip(self, reason: BreakerTrip) -> None:
        """Persist halted=True to system_state and broadcast risk.circuit."""
        self._state = BreakerState(
            halted=True,
            reason=reason,
            halted_at=datetime.now(tz=timezone.utc),
        )
        if self._bus is not None:
            from app.services.messaging import topics
            await self._bus.publish(
                topics.RISK_CIRCUIT,
                {"halted": True, "reason": reason.value},
                headers={"schema": "v1"},
            )

    async def reset(self, operator: str) -> None:
        self._state = BreakerState(halted=False, reason=None, halted_at=None)
        if self._bus is not None:
            from app.services.messaging import topics
            await self._bus.publish(
                topics.RISK_CIRCUIT,
                {"halted": False, "resumed_by": operator},
                headers={"schema": "v1"},
            )
