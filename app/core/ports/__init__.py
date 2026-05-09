"""Port/protocol layer (hexagonal boundary).

Every external dependency (broker, market data, cache, vector store, bus, LLM,
notifier) is invoked through one of these Protocols. Concrete adapters live in
``app/connectors/`` and ``app/services/``. This is the only module the core
business logic is allowed to import for I/O.

Per IMPLEMENTATION_PLAN.md §3.5 + §3.10 Op2 (DI surface) and the parity rule
in BACKTESTING.md §5: backtest replays the same scenarios by swapping these
ports for deterministic in-memory implementations.
"""
from .market_data import MarketDataProvider, Bar, Quote
from .broker import BrokerAdapter, Order, Fill, Position, OrderSide, OrderType
from .macro import MacroDataProvider
from .news import NewsProvider, NewsItem
from .cache import KVCache
from .vector_store import VectorStore, VectorHit
from .bus import MessageBus, BusMessage
from .notifier import Notifier, NotificationChannel
from .llm import LLMClient, LLMResponse, ToolSpec, Tool

__all__ = [
    "MarketDataProvider", "Bar", "Quote",
    "BrokerAdapter", "Order", "Fill", "Position", "OrderSide", "OrderType",
    "MacroDataProvider",
    "NewsProvider", "NewsItem",
    "KVCache",
    "VectorStore", "VectorHit",
    "MessageBus", "BusMessage",
    "Notifier", "NotificationChannel",
    "LLMClient", "LLMResponse", "ToolSpec", "Tool",
]
