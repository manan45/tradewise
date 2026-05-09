"""Base consumer — manual ack, exponential backoff on handler errors."""
from __future__ import annotations

import asyncio
import logging

from app.core.ports.bus import BusMessage, MessageBus

logger = logging.getLogger(__name__)

MAX_CONSECUTIVE_FAILURES = 10


class BaseConsumer:
    topic: str = ""
    group: str = ""

    def __init__(self, bus: MessageBus):
        self.bus = bus

    async def handle(self, msg: BusMessage) -> None:
        raise NotImplementedError

    async def run_forever(self) -> None:
        failures = 0
        backoff = 1.0
        while failures < MAX_CONSECUTIVE_FAILURES:
            try:
                async for msg in await self.bus.subscribe(self.topic, self.group):
                    try:
                        await self.handle(msg)
                        failures = 0
                        backoff = 1.0
                    except Exception as exc:
                        failures += 1
                        logger.error("Consumer handler error: %s", exc, exc_info=True)
                        await asyncio.sleep(min(backoff, 60.0))
                        backoff *= 2
                        if failures >= MAX_CONSECUTIVE_FAILURES:
                            raise RuntimeError(
                                f"Consumer {self.__class__.__name__} exceeded max failures"
                            ) from exc
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("Consumer bus error: %s", exc, exc_info=True)
                await asyncio.sleep(min(backoff, 60.0))
                backoff *= 2
