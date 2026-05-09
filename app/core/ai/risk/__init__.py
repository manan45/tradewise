"""Risk gateway — every order intent passes through here before submission.

Responsibilities (final_requirements §12):
- Position sizing (Kelly, capped at 1% per trade).
- Per-asset, per-sector, per-portfolio exposure caps.
- Event blackouts (earnings, FOMC, NFP, CPI, OPEX week).
- Correlation gate (don't open three highly correlated longs).
- Circuit breaker (system_state.trading_halted == true => block all new opens).

The gateway is pure synchronous Python and never makes I/O calls; the caller
prepares snapshots (account equity, open positions, calendar) and passes them
in. This keeps the gate deterministic and replayable in backtests.
"""
from .gateway import RiskGateway, RiskDecision, RiskRejection
from .sizing import kelly_fraction, position_size_shares
from .blackouts import EventBlackout, BlackoutWindow
from .circuit_breaker import CircuitBreaker, BreakerTrip

__all__ = [
    "RiskGateway", "RiskDecision", "RiskRejection",
    "kelly_fraction", "position_size_shares",
    "EventBlackout", "BlackoutWindow",
    "CircuitBreaker", "BreakerTrip",
]
