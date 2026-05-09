"""Internal messaging — bus publishers + subscribers.

Two layers:
- topics.py — string constants for every routing key (avoid typos)
- publisher.py — typed wrappers around MessageBus.publish for each topic
- consumer.py — base class for long-running subscribers with backoff/retry
"""
from .publisher import EventPublisher
from .consumer import BaseConsumer
from . import topics

__all__ = ["EventPublisher", "BaseConsumer", "topics"]
