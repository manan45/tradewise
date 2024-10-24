from app.utils.ai_model import generate_trade_suggestions
from app.core.domain.entities import Stock, DetailedTradeSuggestion
import pandas as pd

class TradeSuggestions:
    """
    Use case for generating trade suggestions based on stock data.
    """

    def __init__(self, stock_repository):
        """
        Initialize with a stock repository.

        :param stock_repository: Repository for accessing stock data.
        """
        self.stock_repository = stock_repository

    def generate_suggestions(self):
        """
        Generate trade suggestions using AI model.

        :return: List of trade suggestions.
        """
        stocks = self.stock_repository.get_all_stocks()
        stock_data = self._prepare_stock_data(stocks)
        suggestions = generate_trade_suggestions(stock_data)
        return suggestions

    def _prepare_stock_data(self, stocks):
        """
        Prepare stock data for AI model.

        :param stocks: List of stock entities.
        :return: DataFrame of stock data.
        """
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
