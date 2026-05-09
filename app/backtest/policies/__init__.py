"""Backtest policies — knob bundles for sweeps.

Each Policy is a frozen config that fully determines a backtest's behaviour
(scenario list, sizing config, breaker thresholds, friction parameters).
Sweeps are produced by combining policies in app/backtest/runners/sweep.py.
"""
from .base import Policy

__all__ = ["Policy"]
