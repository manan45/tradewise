"""SessionRepository — Protocol for persisting Session rows.

Two implementations are expected:
- TimescaleSessionRepository: writes to the `sessions` hypertable.
- InMemorySessionRepository: used by backtest + tests.

The cheaper model wires the Timescale impl in
app/core/repositories/session_repository.py once the SQLAlchemy session is
configured.
"""
from __future__ import annotations

from typing import Iterable, Protocol
from uuid import UUID

from .models import Session, SessionStatus


class SessionRepository(Protocol):
    async def create(self, session: Session) -> None:
        ...

    async def update(self, session: Session) -> None:
        ...

    async def get(self, session_id: UUID) -> Session | None:
        ...

    async def list_open(
        self,
        symbol: str | None = None,
        scenario: str | None = None,
    ) -> list[Session]:
        ...

    async def close(
        self,
        session_id: UUID,
        outcome: dict,
        pnl: float,
        status: SessionStatus = SessionStatus.CLOSED,
    ) -> None:
        ...

    async def find_neighbours(
        self,
        embedding: list[float],
        scenario: str | None = None,
        top_k: int = 8,
    ) -> Iterable[Session]:
        """pgvector nearest-neighbour search over birth_embedding."""
        ...
