"""Scenario library + manager.

A *scenario* is a pluggable detector + planner. ``on_candle`` is called for
every bar in both live and backtest modes (BACKTESTING.md §5 parity rule);
when ``detect`` fires, a Session opens with a structured plan; the plan is
then policed by the risk gateway and (optionally) reviewed by the LLM
orchestrator before submission.
"""
from .base import Scenario, ScenarioContext, ScenarioSignal, SignalKind
from .manager import ScenarioManager

__all__ = [
    "Scenario", "ScenarioContext", "ScenarioSignal", "SignalKind",
    "ScenarioManager",
]
