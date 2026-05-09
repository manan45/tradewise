"""Prometheus metrics registry — singletons created at startup."""
from __future__ import annotations


class MetricsRegistry:
    def __init__(self) -> None:
        self.bars_ingested = None
        self.signals_emitted = None
        self.orders_submitted = None
        self.orders_rejected = None
        self.breaker_trips = None
        self.model_latency_ms = None
        self.feature_build_ms = None

    def init(self) -> None:
        """Construct prometheus_client metrics. Call once at startup."""
        from prometheus_client import Counter, Histogram
        self.bars_ingested = Counter(
            "twise_bars_ingested_total", "Total bars ingested")
        self.signals_emitted = Counter(
            "twise_signals_emitted_total", "Total scenario signals emitted",
            labelnames=["scenario"])
        self.orders_submitted = Counter(
            "twise_orders_submitted_total", "Total orders submitted")
        self.orders_rejected = Counter(
            "twise_orders_rejected_total", "Total orders rejected",
            labelnames=["reason"])
        self.breaker_trips = Counter(
            "twise_breaker_trips_total", "Total circuit breaker trips",
            labelnames=["reason"])
        self.model_latency_ms = Histogram(
            "twise_model_latency_ms", "Model inference latency",
            labelnames=["model"],
            buckets=[1, 5, 10, 25, 50, 100, 250, 500, 1000])
        self.feature_build_ms = Histogram(
            "twise_feature_build_ms", "Feature build latency",
            buckets=[1, 5, 10, 25, 50, 100, 250])


_REGISTRY: MetricsRegistry | None = None


def get_metrics() -> MetricsRegistry:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = MetricsRegistry()
    return _REGISTRY
