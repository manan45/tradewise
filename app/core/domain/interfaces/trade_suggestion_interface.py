from abc import ABC, abstractmethod
from typing import List, Any
from ..entities.stock import Stock
from ..entities.trade_suggestion import DetailedTradeSuggestion


class TradeSuggestionsServiceInterface(ABC):
    @abstractmethod
    async def suggest_trades(self, stocks: List[Stock]) -> List[DetailedTradeSuggestion]:
        """
        Suggest trades based on current stock data.
        
        Args:
            stocks: A list of Stock objects.
        
        Returns:
            A list of DetailedTradeSuggestion objects.
        """
        pass

    @abstractmethod
    def prepare_stock_data(self, stocks: List[Stock]) -> Any:
        """
        Prepare stock data for analysis.
        
        Args:
            stocks: A list of Stock objects.
        
        Returns:
            Prepared stock data.
        """
        pass

    @abstractmethod
    async def fetch_additional_data(self, stocks: List[Stock]) -> Any:
        """
        Fetch additional data for trading strategy.
        
        Args:
            stocks: A list of Stock objects.
        
        Returns:
            Additional data for trading strategy.
        """
        pass

    @abstractmethod
    async def apply_trading_strategy(self, prepared_data: Any) -> List[DetailedTradeSuggestion]:
        """
        Apply trading strategy to prepared data.
        
        Args:
            prepared_data: Prepared data for trading strategy.
        
        Returns:
            A list of DetailedTradeSuggestion objects.
        """
        pass
