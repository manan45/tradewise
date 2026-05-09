"""Event-driven backtest engine (L2/L3).

Pumps bars from the replayer through the ScenarioManager, runs the same
RiskGateway, hands approved orders to the SimBroker, records every fill on
a virtual portfolio, persists session rows with mode='backtest'.
"""
from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import polars as pl

from app.core.ai.scenarios.manager import ScenarioManager
from app.core.ai.scenarios.base import ScenarioContext
from app.core.ai.risk.gateway import RiskGateway
from app.core.ai.risk.sizing import SizingInputs
from app.core.ai.training.metrics import sharpe, max_drawdown, hit_rate, profit_factor
from .replayer import BarReplayer
from .frictions.sim_broker import SimBroker


@dataclass
class BacktestConfig:
    start: date
    end: date
    starting_equity: float = 100_000.0
    universe: list[str] = field(default_factory=list)
    interval: str = "1d"
    seed: int = 42
    persist_sessions: bool = True


@dataclass
class BacktestResult:
    equity_curve: list[tuple[date, float]]
    trades: list[dict]
    metrics: dict[str, float]
    sessions: list[dict]


class BacktestEngine:
    def __init__(
        self,
        replayer: BarReplayer,
        scenarios: ScenarioManager,
        risk: RiskGateway,
        broker: SimBroker,
    ):
        self.replayer = replayer
        self.scenarios = scenarios
        self.risk = risk
        self.broker = broker

    def run(self, config: BacktestConfig) -> BacktestResult:
        return asyncio.get_event_loop().run_until_complete(self._run_async(config))

    async def _run_async(self, config: BacktestConfig) -> BacktestResult:
        random.seed(config.seed)
        np.random.seed(config.seed)

        equity_curve: list[tuple[date, float]] = []
        bar_buffer: dict[str, list] = {}  # symbol -> list of bars for features
        prior_bars: dict[str, object] = {}
        current_prices: dict[str, float] = {}

        async for bar in await self.replayer.stream(config.start, config.end, config.universe):
            symbol = bar.symbol
            current_prices[symbol] = bar.close
            bar_buffer.setdefault(symbol, []).append(bar)

            # Build a minimal features DataFrame from the bar buffer
            buf = bar_buffer[symbol]
            closes = pl.Series("close", [b.close for b in buf])
            highs = pl.Series("high", [b.high for b in buf])
            lows = pl.Series("low", [b.low for b in buf])
            features = pl.DataFrame({
                "close": closes,
                "high": highs,
                "low": lows,
            })

            ctx = ScenarioContext(
                features=features,
                regime=None,
                open_sessions=[],
            )

            signals = self.scenarios.on_candle(symbol, bar.ts, ctx)

            for signal in signals:
                intent = signal.plan
                intent["symbol"] = symbol
                p_win = signal.confidence
                win_loss_ratio = 2.0  # default 2:1 unless plan provides
                if "target" in intent and "entry" in intent and "stop" in intent:
                    entry = intent["entry"]
                    target = intent.get("target", entry)
                    stop = intent.get("stop", entry)
                    if abs(entry - stop) > 0:
                        win_loss_ratio = abs(target - entry) / abs(entry - stop)

                sizing = SizingInputs(p_win=max(p_win, 0.51), win_loss_ratio=max(win_loss_ratio, 0.1))
                positions = await self.broker.list_positions()
                decision = self.risk.evaluate(
                    intent=intent,
                    sizing=sizing,
                    ts=bar.ts,
                    account_equity=self.broker.equity,
                    open_positions=positions,
                    sector_lookup={},
                )
                if decision.approved and decision.order is not None:
                    await self.broker.submit(decision.order)

            # Process fills on current bar
            self.broker.process_bar(bar, prior_bars.get(symbol))
            prior_bars[symbol] = bar

            # Mark to market
            self.broker.mark_to_market(current_prices)

            # Record equity
            total_equity = self.broker.equity + sum(
                p.market_value for p in self.broker._positions.values()
            )
            equity_curve.append((bar.ts.date(), total_equity))

        trades = [
            {
                "order_id": f.order_id,
                "symbol": f.symbol,
                "side": f.side.value,
                "qty": f.qty,
                "price": f.price,
                "ts": f.ts.isoformat(),
                "commission": f.commission,
            }
            for f in self.broker.fill_log
        ]

        eq_vals = np.array([e for _, e in equity_curve], dtype=float)
        rets = np.diff(eq_vals) / eq_vals[:-1] if len(eq_vals) > 1 else np.array([])
        metrics: dict[str, float] = {}
        if len(rets) > 1:
            metrics["sharpe"] = sharpe(rets)
            metrics["max_drawdown"] = max_drawdown(eq_vals)
            metrics["hit_rate"] = hit_rate(rets)
            metrics["profit_factor"] = profit_factor(rets)
            metrics["total_return"] = float((eq_vals[-1] / eq_vals[0]) - 1) if eq_vals[0] > 0 else 0.0

        return BacktestResult(
            equity_curve=equity_curve,
            trades=trades,
            metrics=metrics,
            sessions=[],
        )
