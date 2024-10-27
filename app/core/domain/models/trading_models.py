from sqlalchemy import Column, Integer, String, Float, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class TradingSession(Base):
    __tablename__ = "trading_sessions"
    
    id = Column(String, primary_key=True)
    start_time = Column(DateTime)
    end_time = Column(DateTime)
    symbol = Column(String)
    interval = Column(String)
    total_trades = Column(Integer)
    winning_trades = Column(Integer)
    losing_trades = Column(Integer)
    win_rate = Column(Float)
    avg_profit = Column(Float)
    max_drawdown = Column(Float)
    sharpe_ratio = Column(Float)
    psychological_state = Column(JSON)
    technical_state = Column(JSON)

class Prediction(Base):
    __tablename__ = "predictions"
    
    id = Column(String, primary_key=True)
    session_id = Column(String)
    timestamp = Column(DateTime)
    symbol = Column(String)
    predicted_value = Column(Float)
    actual_value = Column(Float)
    confidence = Column(Float)
    mae = Column(Float)
    mse = Column(Float)
