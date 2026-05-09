"""Session retrieval — pgvector lookups against the `sessions` table.

Used by the LLM tool `search_similar_sessions` and the dashboard's Session
Replay page (VALIDATION.md). Different from the `rag` service: that one
indexes external corpora; this one only indexes our own past trade sessions.
"""
from .session_search import SessionSearch

__all__ = ["SessionSearch"]
