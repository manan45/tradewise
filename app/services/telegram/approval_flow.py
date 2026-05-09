"""Inline-keyboard approve/reject flow on Telegram."""
from __future__ import annotations

import asyncio
import json
from typing import Any

from app.core.ports.cache import KVCache
from app.core.ports.notifier import NotificationChannel, Notifier


class TelegramApprovalFlow:
    def __init__(self, notifier: Notifier, cache: KVCache,
                 chat_id: str, timeout_seconds: int = 300):
        self.notifier = notifier
        self.cache = cache
        self.chat_id = chat_id
        self.timeout_seconds = timeout_seconds
        self._pending: dict[str, asyncio.Future] = {}

    async def request_approval(
        self,
        request_id: str,
        intent: dict[str, Any],
    ) -> dict[str, Any]:
        """Returns {'approved': bool, 'edits': dict | None}."""
        await self.cache.set(
            f"hitl:{request_id}",
            json.dumps(intent).encode(),
            ttl_seconds=self.timeout_seconds,
        )
        summary = ", ".join(f"{k}={v}" for k, v in intent.items())
        body = f"Request ID: {request_id}\n{summary}"
        await self.notifier.send(
            channel=NotificationChannel.TELEGRAM,
            recipient=self.chat_id,
            subject="Trade Approval Required",
            body=body,
        )
        loop = asyncio.get_event_loop()
        future: asyncio.Future = loop.create_future()
        self._pending[request_id] = future
        try:
            result = await asyncio.wait_for(
                asyncio.shield(future), timeout=self.timeout_seconds
            )
            return result
        except asyncio.TimeoutError:
            self._pending.pop(request_id, None)
            return {"approved": False, "edits": None, "reason": "timeout"}

    async def handle_callback(self, request_id: str, choice: str,
                               edits: dict | None = None) -> None:
        future = self._pending.pop(request_id, None)
        if future and not future.done():
            future.set_result({"approved": choice == "yes", "edits": edits})
