"""Key/value cache port (Redis-backed in prod, dict-backed in tests).

Used for hot session state, idempotency keys, rate-limit windows, and the
embedding cache for cheap repeat lookups.
"""
from __future__ import annotations

from typing import Protocol


class KVCache(Protocol):
    async def get(self, key: str) -> bytes | None:
        ...

    async def set(
        self,
        key: str,
        value: bytes,
        ttl_seconds: int | None = None,
    ) -> None:
        ...

    async def delete(self, key: str) -> None:
        ...

    async def incr(self, key: str, ttl_seconds: int | None = None) -> int:
        ...
