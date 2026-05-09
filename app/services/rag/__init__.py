"""RAG service — retrieves filings, prior post-mortems, research notes.

Embedder: BAAI/bge-base-en-v1.5 (768-d). Store: Qdrant collection `rag_corpus`.
The orchestrator's `search_similar_sessions` tool hits this for nearest
historical sessions.
"""
from .embedder import BgeEmbedder
from .indexer import RagIndexer
from .retriever import RagRetriever

__all__ = ["BgeEmbedder", "RagIndexer", "RagRetriever"]
