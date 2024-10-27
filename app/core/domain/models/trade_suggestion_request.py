from pydantic import BaseModel, ConfigDict
from datetime import datetime
from typing import Optional


class TradeSuggestionRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    
    symbol: str
    date: Optional[str] = None
    interval: Optional[str] = "1d"
    lookback_periods: Optional[int] = 100
    include_analysis: Optional[bool] = False
