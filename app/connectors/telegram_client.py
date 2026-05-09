"""Telegram adapter implementing Notifier for the TELEGRAM channel only."""
from __future__ import annotations

import httpx

from app.core.ports.notifier import NotificationChannel


class TelegramClient:
    def __init__(self, bot_token: str):
        self.bot_token = bot_token
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=10.0)
        return self._client

    async def aclose(self) -> None:
        if self._client:
            await self._client.aclose()

    def _escape_md(self, text: str) -> str:
        special = r"\_*[]()~`>#+-=|{}.!"
        return "".join(f"\\{c}" if c in special else c for c in text)

    async def send(
        self,
        channel: NotificationChannel,
        recipient: str,
        subject: str,
        body: str,
        attachments: list[bytes] | None = None,
    ) -> None:
        text = f"*{self._escape_md(subject)}*\n{self._escape_md(body)}"
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id": recipient,
            "text": text,
            "parse_mode": "MarkdownV2",
        }
        client = self._get_client()
        resp = await client.post(url, json=payload)
        resp.raise_for_status()

    async def send_with_keyboard(
        self,
        recipient: str,
        text: str,
        inline_keyboard: list[list[dict]],
    ) -> None:
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id": recipient,
            "text": text,
            "parse_mode": "MarkdownV2",
            "reply_markup": {"inline_keyboard": inline_keyboard},
        }
        client = self._get_client()
        resp = await client.post(url, json=payload)
        resp.raise_for_status()
