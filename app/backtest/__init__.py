"""Backtest engine.

Three tiers (BACKTESTING.md §2):
- L1: vectorbt-based research sweeps (pure NumPy/Pandas, fast).
- L2: event-driven strategy runs through this engine.
- L3: full system replay — runs the same ScenarioManager.on_candle as live,
  with deterministic in-memory ports (SimBroker, in-mem cache, fixture LLM).

The L3 path is the parity guarantee. If L3 backtest equity diverges from
paper trading on the same data, parity is broken and we ship nothing.
"""
from .engine import BacktestEngine, BacktestConfig, BacktestResult
from .replayer import BarReplayer

__all__ = ["BacktestEngine", "BacktestConfig", "BacktestResult", "BarReplayer"]
