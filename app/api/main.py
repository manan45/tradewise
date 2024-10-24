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
from flask import Flask, jsonify
from app.core.interface_adapters.dhan import DhanAPI
from app.core.use_cases.trade_suggestions import TradeSuggestions

app = Flask(__name__)

@app.route('/trade-suggestions', methods=['GET'])
def get_trade_suggestions():
    dhan_api = DhanAPI(api_key="your_api_key")
    trade_suggestions = TradeSuggestions(stock_repository=dhan_api)
    suggestions = trade_suggestions.generate_suggestions()
    return jsonify(suggestions)

if __name__ == '__main__':
    app.run(debug=True)
