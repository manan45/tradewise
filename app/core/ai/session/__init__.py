"""Session lifecycle: every scenario open creates one Session row.

Session state mirrors the `sessions` hypertable (migration a1b2c3d4e5f6).
This module owns the in-process model + repository façade; persistence is
handed off via the SessionRepository interface so backtest mode can swap in
an in-memory implementation.
"""
from .models import Session, SessionStatus, SessionMode
from .repository import SessionRepository

__all__ = ["Session", "SessionStatus", "SessionMode", "SessionRepository"]
