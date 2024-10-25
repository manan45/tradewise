from dataclasses import dataclass, field
from decimal import Decimal
from typing import List, Optional
from datetime import datetime

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

@dataclass
class Index:
    id: Optional[int]
    name: str

@dataclass
class MutualFund:
    id: Optional[int]
    name: str
