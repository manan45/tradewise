from pydantic import BaseModel, Field
from decimal import Decimal

class DetailedTradeSuggestion(BaseModel):
    action: str = Field(...)
    price: Decimal = Field(...)
    confidence: float = Field(...)
    stop_loss: Decimal = Field(...)
    order_limit: Decimal = Field(...)
    max_risk: Decimal = Field(...)
    max_reward: Decimal = Field(...)
    open: Decimal = Field(...)
    high: Decimal = Field(...)
    low: Decimal = Field(...)
    close: Decimal = Field(...)