from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional

@dataclass
class SessionStats:
    """Statistics for a trading session"""
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
    model: Optional[object] = None
    reinforcement_stats: Optional[Dict] = None
    prediction_accuracy: Optional[float] = None
