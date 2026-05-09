"""Morning brief: regime, watchlist, blackouts, last-night news digest."""
from __future__ import annotations

from datetime import date

from app.core.ports.notifier import NotificationChannel, Notifier


class DailyBriefSender:
    def __init__(self, notifier: Notifier, chat_id: str):
        self.notifier = notifier
        self.chat_id = chat_id

    async def send(self, as_of: date, brief: dict) -> None:
        open_sessions = brief.get("open_sessions", 0)
        pnl = brief.get("pnl_usd", 0.0)
        breaker = brief.get("breaker_halted", False)
        top_signals = brief.get("top_signals", [])

        lines = [
            f"Date: {as_of.isoformat()}",
            f"Open Sessions: {open_sessions}",
            f"P&L: ${pnl:,.2f}",
            f"Circuit Breaker: {'HALTED' if breaker else 'OK'}",
        ]
        if top_signals:
            lines.append("Top Signals:")
            for sig in top_signals[:5]:
                lines.append(f"  - {sig}")
        body = "\n".join(lines)
        await self.notifier.send(
            channel=NotificationChannel.TELEGRAM,
            recipient=self.chat_id,
            subject=f"Daily Brief {as_of.isoformat()}",
            body=body,
        )
