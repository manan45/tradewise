from app.utils.ai_model import generate_trade_suggestions
import pandas as pd
from app.core.domain.services.trade_suggestions_service_interface import TradeSuggestionsServiceInterface

class TradeSuggestionsService(TradeSuggestionsServiceInterface):
    def __init__(self, stock_repository):
        self.stock_repository = stock_repository

    async def suggest_trades(self):
        stocks = await self.stock_repository.get_all_stocks()
        stock_data = self._prepare_stock_data(stocks)
        suggestions = generate_trade_suggestions(stock_data)
        return suggestions

    def _prepare_stock_data(self, stocks):
        data = [{'symbol': stock.symbol, 'close': stock.price} for stock in stocks]
        return pd.DataFrame(data)
