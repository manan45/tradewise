"""Message bus port — RabbitMQ in prod (per IMPLEMENTATION_PLAN §3.10 Op4).

Topics: market.bars, signals.scenario_open, signals.scenario_close,
orders.intent, orders.filled, risk.circuit, audit.event.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, AsyncIterator, Protocol


@dataclass(frozen=True)
class BusMessage:
    topic: str
    key: str | None
    payload: dict[str, Any]
    headers: dict[str, str]


class MessageBus(Protocol):
    async def publish(
        self,
        topic: str,
        payload: dict[str, Any],
        key: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        ...

    async def subscribe(
        self,
        topic: str,
        group: str,
    ) -> AsyncIterator[BusMessage]:
        ...
