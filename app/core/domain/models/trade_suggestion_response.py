from pydantic import BaseModel
from typing import List, Dict, Optional
from datetime import datetime

class TradeSuggestion(BaseModel):
    symbol: str
    action: str
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_reward: float
    timeframe: str
    timestamp: datetime
    analysis: Optional[Dict] = None
    signals: List[Dict]
    psychological_state: Dict
    technical_state: Dict
    zone_state: Dict

class SessionStatsResponse(BaseModel):
    session_id: str
    start_time: datetime
    end_time: datetime
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_profit: float
    max_drawdown: float
    sharpe_ratio: float
    psychological_state: Dict
    technical_state: Dict

class PredictionStatsResponse(BaseModel):
    session_id: str
    timestamp: datetime
    mae: float
    mse: float
    accuracy: float
    precision: float
    recall: float
    predictions: List[Dict]
    actual_values: List[float]

class LogEntry(BaseModel):
    timestamp: datetime
    name: str
    level: str
    message: str

class MarketAnalysis(BaseModel):
    technical_analysis: Dict
    psychological_analysis: Dict
    zone_analysis: Dict
    predictions: List[Dict]
    recommendations: List[Dict]
