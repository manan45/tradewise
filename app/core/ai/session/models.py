"""In-process session model — mirrors the sessions table schema."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4


class SessionStatus(str, Enum):
    OPEN = "open"
    PLANNED = "planned"      # plan staged, awaiting trigger
    LIVE = "live"            # has an open position
    CLOSED = "closed"
    ABORTED = "aborted"      # killed by risk gateway / circuit breaker


class SessionMode(str, Enum):
    LIVE = "live"
    PAPER = "paper"
    BACKTEST = "backtest"


@dataclass
class Session:
    id: UUID = field(default_factory=uuid4)
    opened_at: datetime = field(default_factory=lambda: datetime.utcnow())
    closed_at: datetime | None = None
    symbol: str = ""
    scenario: str = ""
    regime: str | None = None
    status: SessionStatus = SessionStatus.OPEN
    mode: SessionMode = SessionMode.LIVE

    # Structured snapshot of the world at session open (final_requirements §8.7).
    birth_state: dict[str, Any] = field(default_factory=dict)
    # 768-d vector for nearest-neighbour priors (BGE-base output).
    birth_embedding: list[float] | None = None

    plan: dict[str, Any] | None = None       # entry/stop/target plan
    outcome: dict[str, Any] | None = None    # post-trade attribution
    pnl: float | None = None
    notes: str | None = None
