from fastapi import FastAPI
from app.core.use_cases.trade_suggestions import TradeSuggestions
from brain.core.interface_adapters.repositories import InMemoryStockRepository
from app.core.domain.entities import Stock

app = FastAPI()

@app.get("/trade-suggestions")
def get_trade_suggestions():
    stock_repo = InMemoryStockRepository([
        Stock(symbol="AAPL", price=150),
        Stock(symbol="GOOGL", price=2800),
    ])
    trade_suggestions = TradeSuggestions(stock_repo)
    return trade_suggestions.get_suggestions()
