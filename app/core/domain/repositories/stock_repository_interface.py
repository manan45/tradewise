from abc import ABC, abstractmethod
from typing import List, Optional
from decimal import Decimal
from ..entities.stock import Stock, StockPrice

class StockRepositoryInterface(ABC):
    @abstractmethod
    async def get_all_stocks(self) -> List[Stock]:
        """
        Retrieve all stocks from the repository.

        Returns:
            A list of Stock objects representing all stocks in the repository.
        """
        pass

    @abstractmethod
    async def get_stock_by_symbol(self, symbol: str) -> Optional[Stock]:
        """
        Retrieve a specific stock by its symbol.

        Args:
            symbol (str): The unique symbol identifier for the stock.

        Returns:
            A Stock object if found, or None if not found.
        """
        pass

    @abstractmethod
    async def update_stock_price(self, symbol: str, price: Decimal) -> bool:
        """
        Update the price of a specific stock.

        Args:
            symbol (str): The unique symbol identifier for the stock.
            price (Decimal): The new price to set for the stock.

        Returns:
            Boolean indicating whether the update was successful.
        """
        pass

    @abstractmethod
    async def add_stock(self, stock: Stock) -> bool:
        """
        Add a new stock to the repository.

        Args:
            stock (Stock): The new stock to add.

        Returns:
            Boolean indicating whether the addition was successful.
        """
        pass

    @abstractmethod
    async def remove_stock(self, symbol: str) -> bool:
        """
        Remove a stock from the repository.

        Args:
            symbol (str): The unique symbol identifier for the stock.

        Returns:
            Boolean indicating whether the removal was successful.
        """
        pass

    @abstractmethod
    async def add_price_history(self, symbol: str, price_data: StockPrice) -> bool:
        """
        Add price history for a specific stock.

        Args:
            symbol (str): The unique symbol identifier for the stock.
            price_data (StockPrice): The price history data to add.

        Returns:
            Boolean indicating whether the addition was successful.
        """
        pass

    @abstractmethod
    async def get_price_history(self, symbol: str, days: int = 30) -> List[StockPrice]:
        """
        Retrieve the price history for a specific stock.

        Args:
            symbol (str): The unique symbol identifier for the stock.
            days (int): The number of days to retrieve price history for (default: 30).

        Returns:
            A list of StockPrice objects representing the price history for the stock.
        """
        pass
