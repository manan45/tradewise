"""ScenarioManager — fan-out of bars to every registered scenario.

Same instance runs in live and backtest. The only thing that changes is the
ScenarioContext supplier: live builds it from the feature pipeline + open
sessions cache; backtest builds it from the deterministic replayer.
"""
from __future__ import annotations

from datetime import datetime
from typing import Iterable

from .base import Scenario, ScenarioContext, ScenarioSignal


class ScenarioManager:
    def __init__(self, scenarios: Iterable[Scenario]):
        self._scenarios: list[Scenario] = list(scenarios)
        self._index: dict[str, list[Scenario]] = {}
        self._rebuild_index()

    def _rebuild_index(self) -> None:
        self._index = {}
        for s in self._scenarios:
            for sym in s.applicable_universe():
                self._index.setdefault(sym, []).append(s)

    def register(self, scenario: Scenario) -> None:
        self._scenarios.append(scenario)
        self._rebuild_index()

    def applicable(self, symbol: str) -> list[Scenario]:
        """Return scenarios whose universe includes `symbol`."""
        return self._index.get(symbol, [])

    def on_candle(
        self,
        symbol: str,
        ts: datetime,
        ctx: ScenarioContext,
    ) -> list[ScenarioSignal]:
        """Run every applicable scenario; collect non-None signals."""
        signals: list[ScenarioSignal] = []
        for scenario in self.applicable(symbol):
            signal = scenario.on_candle(symbol, ts, ctx)
            if signal is not None:
                signals.append(signal)
        return signals
