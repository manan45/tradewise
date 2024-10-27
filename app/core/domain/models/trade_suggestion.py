from pydantic import BaseModel, ConfigDict
from typing import Dict
from decimal import Decimal

class DetailedTradeSuggestion(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    Suggestion: str
    Action: str
    Summary: Dict[str, str]
    Risk_Management: Dict[str, str]
    Technical_Analysis: Dict[str, str]
    Forecast_Time: str