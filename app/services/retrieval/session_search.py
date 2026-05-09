"""Find historical sessions whose birth_state most resembles a query."""
from __future__ import annotations

from typing import Any

from app.core.ai.session.repository import SessionRepository


class SessionSearch:
    def __init__(self, sessions: SessionRepository):
        self.sessions = sessions

    async def by_birth_state(
        self,
        embedding: list[float],
        scenario: str | None = None,
        only_closed: bool = True,
        top_k: int = 8,
    ) -> list[dict[str, Any]]:
        filters = {}
        if scenario is not None:
            filters["scenario"] = scenario
        if only_closed:
            filters["status"] = "closed"
        neighbours = await self.sessions.find_neighbours(
            embedding=embedding,
            k=top_k,
            filters=filters,
        )
        return [n.__dict__ if hasattr(n, "__dict__") else n for n in neighbours]
