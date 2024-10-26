from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from sqlalchemy import Column, Integer, String, Float, DateTime
from .base import Base

@dataclass
class Index:
    symbol: str
    name: str
    last_price: Decimal
    change: Decimal
    change_percent: float
    open: Decimal
    high: Decimal
    low: Decimal
    prev_close: Decimal
    timestamp: datetime

class IndexModel(Base):
    __tablename__ = 'indices'

    id = Column(Integer, primary_key=True)
    symbol = Column(String, nullable=False, unique=True)
    name = Column(String, nullable=False)
    last_price = Column(Float, nullable=False)
    change = Column(Float, nullable=False)
    change_percent = Column(Float, nullable=False)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    prev_close = Column(Float, nullable=False)
    timestamp = Column(DateTime, nullable=False)
