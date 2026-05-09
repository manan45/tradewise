"""Append-only audit log — every decision the system makes is recorded.

Storage: JSON-lines on disk + indexed in Postgres for query. The compliance
guarantee is reproducibility: given an audit row, we can reconstruct the
exact features, model versions, and prompt that produced the decision.
"""
from .sink import AuditSink, AuditEvent

__all__ = ["AuditSink", "AuditEvent"]
