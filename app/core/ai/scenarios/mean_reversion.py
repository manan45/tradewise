"""Mean-reversion scenario.

Setup: range-bound regime (model A == 'range'), price > 2 sigma from 20-day
mean, RSI(2) > 95 (short) or < 5 (long). Trigger: counter-trend candle close
back inside the band. Exit: revert to 20-day mean OR time-stop at 5 bars.

Plan (cheaper-model wiring):
- Skip when ctx.regime == 'trend' or 'high_vol_expansion'.
- Required features: 'sma_20', 'std_20', 'rsi_2'.
- Plan: {'side': inferred from sign of deviation, 'entry':last_close,
  'stop': mean +/- 3*std, 'target': mean, 'horizon_days': 5}.
"""
from __future__ import annotations

from datetime import datetime

from .base import Scenario, ScenarioContext, ScenarioSignal, SignalKind


class MeanReversionScenario(Scenario):
    name = "mean_reversion"

    def __init__(self, universe: list[str]):
        self._universe = universe

    def applicable_universe(self) -> list[str]:
        return self._universe

    def on_candle(
        self,
        symbol: str,
        ts: datetime,
        ctx: ScenarioContext,
    ) -> ScenarioSignal | None:
        if ctx.regime in ("trend", "high_vol_expansion"):
            return None

        f = ctx.features
        if f.is_empty():
            return None

        row = f.row(-1, named=True)
        sma_20 = row.get("sma_20")
        std_20 = row.get("std_20")
        rsi_2 = row.get("rsi_2")
        close = row.get("close")

        if any(v is None for v in [sma_20, std_20, rsi_2, close]):
            return None
        if std_20 == 0:
            return None

        deviation = (close - sma_20) / std_20
        long_signal = rsi_2 < 5 and deviation < -2.0
        short_signal = rsi_2 > 95 and deviation > 2.0

        if not (long_signal or short_signal):
            return None

        side = "buy" if long_signal else "sell"
        stop_dist = 3.0 * std_20
        plan = {
            "side": side,
            "symbol": symbol,
            "entry": close,
            "stop": (sma_20 - stop_dist) if long_signal else (sma_20 + stop_dist),
            "target": sma_20,
            "horizon_days": 5,
        }
        return ScenarioSignal(
            kind=SignalKind.OPEN,
            symbol=symbol,
            ts=ts,
            confidence=1.0,
            plan=plan,
            rationale={"sma_20": sma_20, "std_20": std_20, "rsi_2": rsi_2,
                       "deviation_sigma": deviation},
        )
