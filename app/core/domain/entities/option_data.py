from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from .base import Base

@dataclass
class OptionChain:
    symbol: str
    expiry_date: datetime
    strike_price: Decimal
    option_type: str
    last_price: Decimal
    change: Decimal
    volume: int
    open_interest: int
    implied_volatility: float

@dataclass
class OpenInterest:
    symbol: str
    expiry_date: datetime
    strike_price: Decimal
    option_type: str
    open_interest: int
    change_in_oi: int
    timestamp: datetime

class OptionChainModel(Base):
    __tablename__ = 'option_chains'

    id = Column(Integer, primary_key=True)
    symbol = Column(String, nullable=False)
    expiry_date = Column(DateTime, nullable=False)
    strike_price = Column(Float, nullable=False)
    option_type = Column(String, nullable=False)
    last_price = Column(Float, nullable=False)
    change = Column(Float, nullable=False)
    volume = Column(Integer, nullable=False)
    open_interest = Column(Integer, nullable=False)
    implied_volatility = Column(Float, nullable=False)

class OpenInterestModel(Base):
    __tablename__ = 'open_interests'

    id = Column(Integer, primary_key=True)
    symbol = Column(String, nullable=False)
    expiry_date = Column(DateTime, nullable=False)
    strike_price = Column(Float, nullable=False)
    option_type = Column(String, nullable=False)
    open_interest = Column(Integer, nullable=False)
    change_in_oi = Column(Integer, nullable=False)
    timestamp = Column(DateTime, nullable=False)
