"""Dependency-injection container — single place that wires concrete adapters
to the protocols defined in app/core/ports.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class Mode(str, Enum):
    LIVE = "live"
    PAPER = "paper"
    BACKTEST = "backtest"


@dataclass
class Container:
    mode: Mode = Mode.PAPER

    # Ports — populated by build()
    market_data: Any = None       # MarketDataProvider
    broker: Any = None            # BrokerAdapter
    macro: Any = None             # MacroDataProvider
    news: Any = None              # NewsProvider
    cache: Any = None             # KVCache
    vector_store: Any = None      # VectorStore
    bus: Any = None               # MessageBus
    notifier: Any = None          # Notifier
    llm: Any = None               # LLMClient

    # Domain singletons
    sessions: Any = None          # SessionRepository
    risk: Any = None              # RiskGateway
    scenarios: Any = None         # ScenarioManager
    orchestrator: Any = None      # LLMOrchestrator
    publisher: Any = None         # EventPublisher

    @classmethod
    def build(cls, mode: Mode) -> "Container":
        from app.config.settings import settings
        from app.core.ai.risk.blackouts import EventBlackout
        from app.core.ai.risk.circuit_breaker import CircuitBreaker
        from app.core.ai.risk.gateway import RiskGateway
        from app.core.ai.scenarios.manager import ScenarioManager
        from app.core.ai.scenarios.trend_continuation import TrendContinuationScenario
        from app.core.ai.scenarios.mean_reversion import MeanReversionScenario
        from app.core.ai.scenarios.vol_expansion import VolExpansionScenario
        from app.services.messaging.publisher import EventPublisher

        c = cls(mode=mode)

        if mode == Mode.BACKTEST:
            # In-memory adapters for backtest
            from app.backtest.frictions.sim_broker import SimBroker
            from app.backtest.frictions.fills import FillModel
            from app.backtest.frictions.commissions import CommissionModel
            from app.backtest.frictions.slippage import EquitySlippage

            c.broker = SimBroker(
                starting_equity=100_000.0,
                slippage=EquitySlippage(),
                commissions=CommissionModel(),
                fills=FillModel(),
            )
            c.cache = _InMemoryCache()
            c.bus = _InMemoryBus()
            c.notifier = _NoopNotifier()
            c.market_data = None  # BarReplayer is wired separately in engine
        else:
            # Live / Paper adapters
            from app.connectors.polygon_client import PolygonClient
            from app.connectors.alpaca_client import AlpacaBroker, AlpacaMarketData
            from app.connectors.redis_client import RedisClient
            from app.connectors.qdrant_client import QdrantClientAdapter
            from app.connectors.rabbitmq_client import RabbitMQClient
            from app.connectors.telegram_client import TelegramClient
            from app.connectors.fred_client import FredClient

            c.market_data = PolygonClient(api_key=settings.POLYGON_API_KEY)
            paper = (mode == Mode.PAPER)
            c.broker = AlpacaBroker(
                key_id=settings.APCA_API_KEY_ID,
                secret=settings.APCA_API_SECRET_KEY,
                paper=paper,
            )
            c.cache = RedisClient(url=settings.REDIS_URL)
            c.vector_store = QdrantClientAdapter(
                host=settings.QDRANT_HOST, port=settings.QDRANT_PORT
            )
            c.bus = RabbitMQClient(url=settings.RABBITMQ_URL)
            c.notifier = TelegramClient(bot_token=settings.TELEGRAM_BOT_TOKEN)
            c.macro = FredClient(api_key=settings.FRED_API_KEY)

            if not settings.NO_NETWORK:
                from app.services.llm.anthropic_client import AnthropicLLMClient
                c.llm = AnthropicLLMClient(api_key=settings.ANTHROPIC_API_KEY)

        # Wire domain singletons (mode-agnostic)
        breaker = CircuitBreaker()
        breaker._bus = c.bus
        blackouts = EventBlackout([])
        c.risk = RiskGateway(breaker=breaker, blackouts=blackouts)

        universe: list[str] = []  # populated at runtime from universe_membership
        c.scenarios = ScenarioManager([
            TrendContinuationScenario(universe=universe),
            MeanReversionScenario(universe=universe),
            VolExpansionScenario(universe=universe),
        ])

        c.publisher = EventPublisher(bus=c.bus)

        if c.llm is not None:
            from app.services.llm.orchestrator import LLMOrchestrator
            c.orchestrator = LLMOrchestrator(client=c.llm, tools=[])

        return c


class _InMemoryCache:
    def __init__(self) -> None:
        self._store: dict[str, bytes] = {}

    async def get(self, key: str) -> bytes | None:
        return self._store.get(key)

    async def set(self, key: str, value: bytes, ttl_seconds: int | None = None) -> None:
        self._store[key] = value

    async def delete(self, key: str) -> None:
        self._store.pop(key, None)

    async def incr(self, key: str, ttl_seconds: int | None = None) -> int:
        val = int(self._store.get(key, b"0"))
        val += 1
        self._store[key] = str(val).encode()
        return val


class _InMemoryBus:
    async def publish(self, topic: str, payload, key=None, headers=None) -> None:
        pass

    async def subscribe(self, topic: str, group: str):
        if False:
            yield  # pragma: no cover


class _NoopNotifier:
    async def send(self, channel, recipient, subject, body, attachments=None) -> None:
        pass
