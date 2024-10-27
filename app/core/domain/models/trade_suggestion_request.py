from pydantic import BaseModel, ConfigDict


class TradeSuggestionRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    symbol: str
    date: str