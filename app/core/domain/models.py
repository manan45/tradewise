from pydantic.v1 import BaseModel, Field


class DetailedTradeSuggestion(BaseModel):
    action: str = Field(...)
    price: float = Field(...)
    confidence: float = Field(...)
    stop_loss: float = Field(...)
    order_limit: float = Field(...)
    max_risk: float = Field(...)
    max_reward: float = Field(...)
    open: float = Field(...)
    high: float = Field(...)
    low: float = Field(...)
    close: float = Field(...)
