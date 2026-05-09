"""Typed publisher facade — every domain event has a method here."""
from __future__ import annotations

from typing import Any

from app.core.ports.bus import MessageBus
from . import topics


class EventPublisher:
    def __init__(self, bus: MessageBus):
        self.bus = bus

    async def signal_open(self, payload: dict[str, Any]) -> None:
        await self.bus.publish(topics.SIGNAL_OPEN, payload, headers={"schema": "v1"})

    async def signal_close(self, payload: dict[str, Any]) -> None:
        await self.bus.publish(topics.SIGNAL_CLOSE, payload, headers={"schema": "v1"})

    async def order_intent(self, payload: dict[str, Any]) -> None:
        await self.bus.publish(topics.ORDER_INTENT, payload, headers={"schema": "v1"})

    async def order_filled(self, payload: dict[str, Any]) -> None:
        await self.bus.publish(topics.ORDER_FILLED, payload, headers={"schema": "v1"})

    async def risk_decision(self, payload: dict[str, Any]) -> None:
        await self.bus.publish(topics.RISK_DECISION, payload, headers={"schema": "v1"})

    async def circuit(self, payload: dict[str, Any]) -> None:
        await self.bus.publish(topics.RISK_CIRCUIT, payload, headers={"schema": "v1"})

    async def audit(self, payload: dict[str, Any]) -> None:
        await self.bus.publish(topics.AUDIT_EVENT, payload, headers={"schema": "v1"})
