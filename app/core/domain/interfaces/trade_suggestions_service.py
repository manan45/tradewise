from abc import ABC, abstractmethod
from typing import List, Decimal
from ..entities.stock import Stock
from ..entities.trade_suggestion import DetailedTradeSuggestion

class TradeSuggestionsService(ABC):
    @abstractmethod
    async def get_trade_suggestions(self, stocks: List[Stock]) -> List[DetailedTradeSuggestion]:
        """
        Get trade suggestions based on the provided stock data.

        Args:
            stocks: A list of Stock objects.

        Returns:
            A list of DetailedTradeSuggestion objects.
        """
        pass

    @abstractmethod
    async def analyze_stock(self, stock: Stock) -> DetailedTradeSuggestion:
        pass

    @abstractmethod
    async def calculate_confidence(self, stock: Stock) -> float:
        pass

    @abstractmethod
    async def determine_stop_loss(self, stock: Stock) -> Decimal:
        pass

    @abstractmethod
    async def determine_order_limit(self, stock: Stock) -> Decimal:
        pass

    @abstractmethod
    async def calculate_risk_reward(self, stock: Stock, stop_loss: Decimal, order_limit: Decimal) -> tuple[Decimal, Decimal]:
        pass
