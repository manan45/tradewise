from sqlalchemy import Column, Integer, String, Float, DateTime, Enum, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
import enum
from datetime import datetime

Base = declarative_base()

class TradeAction(enum.Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"

class Timeframe(enum.Enum):
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"
    W1 = "1w"

class TradingSession(Base):
    __tablename__ = "trading_sessions"

    id = Column(String(50), primary_key=True)
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime)
    symbol = Column(String(20), nullable=False)
    interval = Column(Enum(Timeframe), nullable=False)
    total_trades = Column(Integer, default=0)
    winning_trades = Column(Integer, default=0)
    losing_trades = Column(Integer, default=0)
    win_rate = Column(Float)
    avg_profit = Column(Float)
    max_drawdown = Column(Float)
    sharpe_ratio = Column(Float)
    psychological_state = Column(JSON)
    technical_state = Column(JSON)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(String(50), primary_key=True)
    session_id = Column(String(50), ForeignKey('trading_sessions.id'))
    timestamp = Column(DateTime, nullable=False)
    symbol = Column(String(20), nullable=False)
    predicted_value = Column(Float, nullable=False)
    actual_value = Column(Float)
    confidence = Column(Float)
    mae = Column(Float)
    mse = Column(Float)
    created_at = Column(DateTime, server_default=func.now())

class TradeSuggestion(Base):
    __tablename__ = "trade_suggestions"

    id = Column(String(50), primary_key=True)
    session_id = Column(String(50), ForeignKey('trading_sessions.id'))
    symbol = Column(String(20), nullable=False)
    action = Column(Enum(TradeAction), nullable=False)
    confidence = Column(Float, nullable=False)
    entry_price = Column(Float, nullable=False)
    stop_loss = Column(Float, nullable=False)
    take_profit = Column(Float, nullable=False)
    risk_reward = Column(Float, nullable=False)
    timeframe = Column(Enum(Timeframe), nullable=False)
    timestamp = Column(DateTime, nullable=False)
    analysis = Column(JSON)
    signals = Column(JSON)
    psychological_state = Column(JSON)
    technical_state = Column(JSON)
    zone_state = Column(JSON)
    created_at = Column(DateTime, server_default=func.now())

class SessionLog(Base):
    __tablename__ = "session_logs"

    id = Column(Integer, primary_key=True)
    session_id = Column(String(50), ForeignKey('trading_sessions.id'))
    timestamp = Column(DateTime, nullable=False)
    level = Column(String(10), nullable=False)
    message = Column(String, nullable=False)
    metadata = Column(JSON)
    created_at = Column(DateTime, server_default=func.now())

class MarketAnalysis(Base):
    __tablename__ = "market_analysis"

    id = Column(String(50), primary_key=True)
    symbol = Column(String(20), nullable=False)
    timestamp = Column(DateTime, nullable=False)
    technical_analysis = Column(JSON)
    psychological_analysis = Column(JSON)
    zone_analysis = Column(JSON)
    predictions = Column(JSON)
    recommendations = Column(JSON)
    created_at = Column(DateTime, server_default=func.now())
