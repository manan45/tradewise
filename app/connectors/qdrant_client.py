"""Qdrant adapter implementing VectorStore."""
from __future__ import annotations

from typing import Any, Sequence

from app.core.ports.vector_store import VectorHit


class QdrantClientAdapter:
    def __init__(self, host: str = "qdrant", port: int = 6334):
        self.host = host
        self.port = port
        self._client = None

    def _get_client(self):
        if self._client is None:
            from qdrant_client import AsyncQdrantClient
            self._client = AsyncQdrantClient(host=self.host, port=self.port, prefer_grpc=True)
        return self._client

    async def upsert(
        self,
        collection: str,
        ids: Sequence[str],
        vectors: Sequence[Sequence[float]],
        payloads: Sequence[dict[str, Any]],
    ) -> None:
        from qdrant_client.models import VectorParams, Distance, PointStruct
        client = self._get_client()
        # Auto-create collection if missing
        collections = await client.get_collections()
        existing = {c.name for c in collections.collections}
        if collection not in existing:
            dim = len(vectors[0]) if vectors else 768
            await client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )
        points = [
            PointStruct(id=id_, vector=list(vec), payload=pay)
            for id_, vec, pay in zip(ids, vectors, payloads)
        ]
        await client.upsert(collection_name=collection, points=points)

    async def search(
        self,
        collection: str,
        query: Sequence[float],
        top_k: int = 8,
        filters: dict[str, Any] | None = None,
    ) -> list[VectorHit]:
        from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny
        client = self._get_client()
        qdrant_filter = None
        if filters:
            conditions = []
            for key, val in filters.items():
                if isinstance(val, list):
                    conditions.append(FieldCondition(key=key, match=MatchAny(any=val)))
                else:
                    conditions.append(FieldCondition(key=key, match=MatchValue(value=val)))
            qdrant_filter = Filter(must=conditions)
        results = await client.search(
            collection_name=collection,
            query_vector=list(query),
            limit=top_k,
            query_filter=qdrant_filter,
        )
        return [
            VectorHit(id=str(r.id), score=r.score, payload=r.payload or {})
            for r in results
        ]
