"""Redis adapter implementing KVCache."""
from __future__ import annotations

import redis.asyncio as aioredis


class RedisClient:
    def __init__(self, url: str):
        self.url = url
        self._redis: aioredis.Redis | None = None

    def _r(self) -> aioredis.Redis:
        if self._redis is None:
            self._redis = aioredis.Redis.from_url(self.url)
        return self._redis

    async def get(self, key: str) -> bytes | None:
        return await self._r().get(key)

    async def set(
        self,
        key: str,
        value: bytes,
        ttl_seconds: int | None = None,
    ) -> None:
        await self._r().set(key, value, ex=ttl_seconds)

    async def delete(self, key: str) -> None:
        await self._r().delete(key)

    async def incr(self, key: str, ttl_seconds: int | None = None) -> int:
        r = self._r()
        val = await r.incr(key)
        if ttl_seconds is not None and val == 1:
            await r.expire(key, ttl_seconds)
        return int(val)

    async def aclose(self) -> None:
        if self._redis is not None:
            await self._redis.aclose()
