"""Monitoring — Prometheus metrics + structured logs + health endpoints.

Exposes /metrics (scraped by infra/prometheus) and a small set of named
counters/histograms for: bars ingested, signals emitted, orders submitted,
orders rejected by reason, risk decisions, breaker trips, model latency.
"""
from .metrics import MetricsRegistry, get_metrics
from .logging import configure_logging

__all__ = ["MetricsRegistry", "get_metrics", "configure_logging"]
