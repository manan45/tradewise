"""Bulk indexer — chunks documents, embeds, upserts to Qdrant."""
from __future__ import annotations

import hashlib

from app.core.ports.vector_store import VectorStore
from .embedder import BgeEmbedder


class RagIndexer:
    def __init__(self, embedder: BgeEmbedder, store: VectorStore,
                 collection: str = "rag_corpus"):
        self.embedder = embedder
        self.store = store
        self.collection = collection

    async def index_documents(
        self,
        docs: list[dict],
        chunk_size: int = 512,
        chunk_overlap: int = 64,
    ) -> int:
        chunks: list[str] = []
        payloads: list[dict] = []
        ids: list[str] = []

        for doc in docs:
            text = doc.get("text", "")
            words = text.split()
            step = max(1, chunk_size - chunk_overlap)
            for i in range(0, max(1, len(words)), step):
                chunk = " ".join(words[i: i + chunk_size])
                if not chunk:
                    continue
                chunks.append(chunk)
                payloads.append({
                    "source": doc.get("source", ""),
                    "ts": doc.get("ts", ""),
                    "symbol": doc.get("symbol", ""),
                    "text": chunk,
                })
                ids.append(hashlib.sha256(chunk.encode()).hexdigest())

        if not chunks:
            return 0

        vectors = await self.embedder.embed(chunks)
        await self.store.upsert(self.collection, ids, vectors, payloads)
        return len(chunks)
