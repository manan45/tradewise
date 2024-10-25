from pydantic import BaseModel


class TradeSuggestionRequest(BaseModel):
    symbol: str
    date: str