"""Vector store port — Qdrant in prod, in-memory FAISS-style stub for tests.

Backs nearest-neighbour priors over the sessions table (final_requirements
§8.7) and the RAG corpus (filings, research notes, prior post-mortems).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence


@dataclass(frozen=True)
class VectorHit:
    id: str
    score: float
    payload: dict[str, Any]


class VectorStore(Protocol):
    async def upsert(
        self,
        collection: str,
        ids: Sequence[str],
        vectors: Sequence[Sequence[float]],
        payloads: Sequence[dict[str, Any]],
    ) -> None:
        ...

    async def search(
        self,
        collection: str,
        query: Sequence[float],
        top_k: int = 8,
        filters: dict[str, Any] | None = None,
    ) -> list[VectorHit]:
        ...
