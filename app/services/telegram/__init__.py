"""Telegram HITL — daily brief delivery + approve/reject inline keyboards.

Approval flow: when an order intent has confidence below auto-execute threshold
or sizing exceeds a policy cap, the gateway emits hitl.request; this service
formats the message with inline buttons and waits (with a 5-min timeout) for
a callback before resolving the original intent.
"""
from .approval_flow import TelegramApprovalFlow
from .daily_brief import DailyBriefSender

__all__ = ["TelegramApprovalFlow", "DailyBriefSender"]
