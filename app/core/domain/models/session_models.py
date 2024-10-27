from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional, Any

@dataclass
class SessionStats:
    """Statistics for a trading session"""
    session_id: str
    start_time: datetime
    end_time: datetime
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    avg_profit: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    psychological_state: Dict = field(default_factory=dict)
    technical_state: Dict = field(default_factory=dict)
    reinforcement_stats: Dict = field(default_factory=lambda: {
        'episode_rewards': [],
        'total_rewards': 0,
        'avg_reward': 0,
        'max_reward': float('-inf'),
        'min_reward': float('inf')
    })
    final_balance: Optional[float] = None
    model: Optional[Any] = None
