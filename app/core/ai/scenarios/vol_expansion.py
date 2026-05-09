"""Vol-expansion scenario (replaces the old IV-mispricing options play).

Setup: realized-vol percentile in bottom decile across the last 60 sessions
(coil) AND model B' fires (RV expansion probability > 0.6). Trigger: range
break of the consolidation high/low with volume confirmation. Exit: 1.5x the
recent ATR or time-stop at 8 bars.

Plan (cheaper-model wiring):
- Required features: 'realized_vol_20', 'rv_percentile_60', 'b_prime_prob',
  'donchian_high_20', 'donchian_low_20', 'atr_14', 'volume_z_20'.
- Direction is whichever side breaks first; never assume long-only.
"""
from __future__ import annotations

from datetime import datetime

from .base import Scenario, ScenarioContext, ScenarioSignal, SignalKind


class VolExpansionScenario(Scenario):
    name = "vol_expansion"

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
        rv_pct_60 = row.get("rv_percentile_60")
        b_prime_prob = row.get("b_prime_prob")
        donchian_high = row.get("donchian_high_20")
        donchian_low = row.get("donchian_low_20")
        atr_14 = row.get("atr_14")
        close = row.get("close")

        if any(v is None for v in [rv_pct_60, b_prime_prob, donchian_high,
                                    donchian_low, atr_14, close]):
            return None

        coil = rv_pct_60 <= 0.10      # bottom decile
        expansion_prob = b_prime_prob >= 0.6

        if not (coil and expansion_prob):
            return None

        # Direction: whichever side closes beyond the donchian channel
        if close > donchian_high:
            side = "buy"
        elif close < donchian_low:
            side = "sell"
        else:
            return None

        plan = {
            "side": side,
            "symbol": symbol,
            "entry": close,
            "stop": (close - 1.5 * atr_14) if side == "buy" else (close + 1.5 * atr_14),
            "target": (close + 1.5 * atr_14) if side == "buy" else (close - 1.5 * atr_14),
            "horizon_days": 8,
        }
        return ScenarioSignal(
            kind=SignalKind.OPEN,
            symbol=symbol,
            ts=ts,
            confidence=b_prime_prob,
            plan=plan,
            rationale={"rv_percentile_60": rv_pct_60, "b_prime_prob": b_prime_prob},
        )
