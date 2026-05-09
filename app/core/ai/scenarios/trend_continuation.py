"""Trend-continuation scenario.

Setup: pullback to the 20EMA inside an established uptrend (50EMA > 200EMA,
ADX > 25, RSI(14) cooled to 40-55). Trigger: bullish reversal candle on the
pullback bar with volume above 20-day median. Exit: trail by 2*ATR(14).

Plan (cheaper-model wiring):
- Pull features 'ema_20', 'ema_50', 'ema_200', 'adx_14', 'rsi_14',
  'atr_14', 'volume_z_20' from the registry.
- Build the plan dict as {'side':'buy', 'entry':last_close,
  'stop':last_close - 2*atr, 'target':last_close + 4*atr,
  'horizon_days': 10}.
"""
from __future__ import annotations

from datetime import datetime

from .base import Scenario, ScenarioContext, ScenarioSignal, SignalKind


class TrendContinuationScenario(Scenario):
    name = "trend_continuation"

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
        f = ctx.features
        if f.is_empty():
            return None

        row = f.row(-1, named=True)
        ema_20 = row.get("ema_20")
        ema_50 = row.get("ema_50")
        ema_200 = row.get("ema_200")
        adx_14 = row.get("adx_14")
        rsi_14 = row.get("rsi_14")
        atr_14 = row.get("atr_14")
        close = row.get("close")

        if any(v is None for v in [ema_20, ema_50, ema_200, adx_14, rsi_14, atr_14, close]):
            return None

        trend_up = ema_50 > ema_200
        pullback = close <= ema_20 * 1.05  # within 5% of 20-day EMA
        adx_strong = adx_14 > 25
        rsi_cooled = 40 <= rsi_14 <= 55

        if not (trend_up and pullback and adx_strong and rsi_cooled):
            return None

        plan = {
            "side": "buy",
            "symbol": symbol,
            "entry": close,
            "stop": close - 2.0 * atr_14,
            "target": close + 4.0 * atr_14,
            "horizon_days": 10,
        }
        return ScenarioSignal(
            kind=SignalKind.OPEN,
            symbol=symbol,
            ts=ts,
            confidence=1.0,
            plan=plan,
            rationale={"ema_20": ema_20, "ema_50": ema_50, "ema_200": ema_200,
                       "adx_14": adx_14, "rsi_14": rsi_14},
        )
