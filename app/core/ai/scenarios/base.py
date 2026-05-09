"""Scenario base class + signal types.

Scenarios are stateless w.r.t. each other; per-symbol working state lives on
the Session. They consume features from the registry (never raw bars) and
emit ScenarioSignal objects that downstream layers turn into orders.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Protocol

import polars as pl


class SignalKind(str, Enum):
    OPEN = "open"        # propose opening a session
    UPDATE = "update"    # adjust stop/target on an open session
    CLOSE = "close"      # exit signal


@dataclass(frozen=True)
class ScenarioSignal:
    kind: SignalKind
    symbol: str
    ts: datetime
    confidence: float
    plan: dict[str, Any] = field(default_factory=dict)  # entry/stop/target/horizon
    rationale: dict[str, Any] = field(default_factory=dict)  # feature snapshot


@dataclass
class ScenarioContext:
    """Bundle of read-only services a scenario needs on each tick."""
    features: pl.DataFrame                       # PIT-correct features for symbol
    regime: str | None                           # latest output of model A
    open_sessions: list[Any]                     # list[Session]; Any to avoid cycle
    extras: dict[str, Any] = field(default_factory=dict)


class Scenario(Protocol):
    name: str

    def applicable_universe(self) -> list[str]:
        """Return the symbols this scenario should be evaluated on."""
        ...

    def on_candle(
        self,
        symbol: str,
        ts: datetime,
        ctx: ScenarioContext,
    ) -> ScenarioSignal | None:
        ...
