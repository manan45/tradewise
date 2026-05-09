"""Twilio adapter implementing Notifier for the SMS channel only."""
from __future__ import annotations

from app.core.ports.notifier import NotificationChannel


class TwilioClient:
    def __init__(self, account_sid: str, auth_token: str, from_number: str):
        self.account_sid = account_sid
        self.auth_token = auth_token
        self.from_number = from_number

    async def send(
        self,
        channel: NotificationChannel,
        recipient: str,
        subject: str,
        body: str,
        attachments: list[bytes] | None = None,
    ) -> None:
        from twilio.rest import Client
        import asyncio
        full_body = f"{subject}: {body}"[:1500]
        client = Client(self.account_sid, self.auth_token)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: client.messages.create(
                body=full_body,
                from_=self.from_number,
                to=recipient,
            ),
        )
