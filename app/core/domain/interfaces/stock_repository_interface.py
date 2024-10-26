from abc import ABC, abstractmethod
from typing import List, Optional
from datetime import datetime
from decimal import Decimal
from ..entities.stock import Stock, StockPrice
from ..entities.option_data import OptionChain, OpenInterest

class StockRepositoryInterface(ABC):
    @abstractmethod
    async def add_stock(self, stock: Stock) -> None:
        pass

    @abstractmethod
    async def get_stock(self, symbol: str) -> Optional[Stock]:
        pass

    @abstractmethod
    async def update_stock(self, stock: Stock) -> None:
        pass

    @abstractmethod
    async def delete_stock(self, symbol: str) -> None:
        pass

    @abstractmethod
    async def add_stock_price(self, stock_price: StockPrice) -> None:
        pass

    @abstractmethod
    async def get_stock_prices(self, symbol: str, start_date: datetime, end_date: datetime) -> List[StockPrice]:
        pass

    @abstractmethod
    async def add_option_chain_data(self, option_chain_data: List[OptionChain]) -> None:
        pass

    @abstractmethod
    async def add_open_interest_data(self, open_interest_data: List[OpenInterest]) -> None:
        pass

    @abstractmethod
    async def get_option_chain_data(self, symbol: str, expiry_date: datetime) -> List[OptionChain]:
        pass

    @abstractmethod
    async def get_open_interest_data(self, symbol: str, expiry_date: datetime, strike_price: Decimal, option_type: str) -> List[OpenInterest]:
        pass

    @abstractmethod
    async def get_latest_option_chain(self, symbol: str) -> List[OptionChain]:
        pass

    @abstractmethod
    async def get_latest_open_interest(self, symbol: str) -> List[OpenInterest]:
        pass
