from dataclasses import dataclass, field
from decimal import Decimal
from typing import List, Optional
from datetime import datetime
from sqlalchemy import Column, Integer, String, Numeric, DateTime, ForeignKey, Sequence
from sqlalchemy.orm import relationship
from .base import Base

@dataclass
class StockPrice:
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int
    timestamp: datetime

@dataclass
class Stock:
    symbol: str
    name: str
    current_price: Decimal
    price_history: List[StockPrice] = field(default_factory=list)
    id: Optional[int] = None

    def update_price(self, new_price: Decimal) -> None:
        self.current_price = new_price

    def add_price_history(self, price_data: StockPrice) -> None:
        self.price_history.append(price_data)

    def get_latest_price(self) -> Optional[StockPrice]:
        return self.price_history[-1] if self.price_history else None

    def calculate_average_price(self, days: int = 30) -> Decimal:
        if not self.price_history or days <= 0:
            return Decimal('0')
        prices = [price.close for price in self.price_history[-days:]]
        return sum(prices) / len(prices)

class StockModel(Base):
    __tablename__ = 'stocks'

    id = Column(Integer, Sequence('stocks_id_seq'), primary_key=True)
    symbol = Column(String, unique=True, nullable=False)
    name = Column(String, nullable=False)
    current_price = Column(Numeric(10, 2), nullable=False)

    price_history = relationship("StockPriceModel", back_populates="stock")

class StockPriceModel(Base):
    __tablename__ = 'stock_prices'

    id = Column(Integer, Sequence('stock_prices_id_seq'), primary_key=True)
    stock_symbol = Column(String, ForeignKey('stocks.symbol'), nullable=False)
    open = Column(Numeric(10, 2), nullable=False)
    high = Column(Numeric(10, 2), nullable=False)
    low = Column(Numeric(10, 2), nullable=False)
    close = Column(Numeric(10, 2), nullable=False)
    volume = Column(Integer, nullable=False)
    timestamp = Column(DateTime, nullable=False)

    stock = relationship("StockModel", back_populates="price_history")

@dataclass
class Index:
    id: Optional[int]
    name: str

@dataclass
class MutualFund:
    id: Optional[int]
    name: str
