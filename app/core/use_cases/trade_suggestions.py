from app.utils.ai_model import generate_trade_suggestions
from app.core.domain.entities import Stock, DetailedTradeSuggestion
from app.utils.ai_model import generate_trade_suggestions
import pandas as pd

class TradeSuggestions:
    def __init__(self, stock_repository):
        self.stock_repository = stock_repository

    def generate_suggestions(self):
        stocks = self.stock_repository.get_all_stocks()
        stock_data = self._prepare_stock_data(stocks)
        suggestions = generate_trade_suggestions(stock_data)
        return suggestions

    def _prepare_stock_data(self, stocks):
        # Convert stocks to a DataFrame or suitable format for the AI model
        data = [{'symbol': stock.symbol, 'close': stock.price} for stock in stocks]
        return pd.DataFrame(data)
class TradeSuggestions:
    def __init__(self, stock_repository):
        self.stock_repository = stock_repository

    def generate_suggestions(self):
        stocks = self.stock_repository.get_all_stocks()
        stock_data = self._prepare_stock_data(stocks)
        suggestions = generate_trade_suggestions(stock_data)
        return suggestions

    def _prepare_stock_data(self, stocks):
        # Convert stocks to a DataFrame or suitable format for the AI model
        data = [{'symbol': stock.symbol, 'close': stock.price} for stock in stocks]
        return pd.DataFrame(data)
