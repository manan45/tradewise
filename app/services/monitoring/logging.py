"""structlog configuration — JSON in prod, key=value in dev."""
from __future__ import annotations

import logging


def configure_logging(level: str = "INFO", json_output: bool = True) -> None:
    """Wire structlog + stdlib logging together. Call once at startup."""
    import structlog

    level_int = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(level=level_int)

    renderer = structlog.processors.JSONRenderer() if json_output else structlog.dev.ConsoleRenderer()

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.ExceptionRenderer(),
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level_int),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
