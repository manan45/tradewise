"""Top-K retriever with optional metadata filters."""
from __future__ import annotations

from typing import Any

from app.core.ports.vector_store import VectorHit, VectorStore
from .embedder import BgeEmbedder


class RagRetriever:
    def __init__(self, embedder: BgeEmbedder, store: VectorStore,
                 collection: str = "rag_corpus"):
        self.embedder = embedder
        self.store = store
        self.collection = collection

    async def search(
        self,
        query: str,
        top_k: int = 8,
        filters: dict[str, Any] | None = None,
    ) -> list[VectorHit]:
        vectors = await self.embedder.embed([query])
        return await self.store.search(self.collection, vectors[0], top_k=top_k, filters=filters)
