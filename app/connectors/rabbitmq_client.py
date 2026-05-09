"""RabbitMQ adapter implementing MessageBus."""
from __future__ import annotations

import json
from typing import Any, AsyncIterator

import aio_pika

from app.core.ports.bus import BusMessage


class RabbitMQClient:
    def __init__(self, url: str, exchange: str = "traderwise.events"):
        self.url = url
        self.exchange = exchange
        self._connection: aio_pika.abc.AbstractConnection | None = None
        self._channel: aio_pika.abc.AbstractChannel | None = None
        self._exchange_obj: aio_pika.abc.AbstractExchange | None = None

    async def _ensure(self) -> tuple[aio_pika.abc.AbstractChannel, aio_pika.abc.AbstractExchange]:
        if self._connection is None or self._connection.is_closed:
            self._connection = await aio_pika.connect_robust(self.url)
        if self._channel is None or self._channel.is_closed:
            self._channel = await self._connection.channel()
            self._exchange_obj = await self._channel.declare_exchange(
                self.exchange,
                aio_pika.ExchangeType.TOPIC,
                durable=True,
            )
        return self._channel, self._exchange_obj  # type: ignore[return-value]

    async def publish(
        self,
        topic: str,
        payload: dict[str, Any],
        key: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        _, exchange = await self._ensure()
        body = json.dumps(payload).encode()
        message = aio_pika.Message(
            body=body,
            delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
            headers=headers or {},
        )
        await exchange.publish(message, routing_key=topic)

    async def subscribe(
        self,
        topic: str,
        group: str,
    ) -> AsyncIterator[BusMessage]:
        channel, exchange = await self._ensure()
        queue = await channel.declare_queue(
            f"{topic}.{group}",
            durable=True,
        )
        await queue.bind(exchange, routing_key=topic)
        async with queue.iterator() as it:
            async for message in it:
                async with message.process():
                    payload = json.loads(message.body.decode())
                    headers = {k: str(v) for k, v in (message.headers or {}).items()}
                    yield BusMessage(
                        topic=topic,
                        key=None,
                        payload=payload,
                        headers=headers,
                    )

    async def aclose(self) -> None:
        if self._connection and not self._connection.is_closed:
            await self._connection.close()
