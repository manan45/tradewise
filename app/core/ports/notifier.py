"""Outbound notifier port — Telegram, Twilio (SMS), and audit log.

Used by HITL approvals (final_requirements §11) and circuit-breaker alerts.
"""
from __future__ import annotations

from enum import Enum
from typing import Protocol


class NotificationChannel(str, Enum):
    TELEGRAM = "telegram"
    SMS = "sms"
    EMAIL = "email"
    AUDIT_LOG = "audit_log"


class Notifier(Protocol):
    async def send(
        self,
        channel: NotificationChannel,
        recipient: str,
        subject: str,
        body: str,
        attachments: list[bytes] | None = None,
    ) -> None:
        ...
