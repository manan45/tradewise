from app.core.domain.interfaces.trade_suggestions_service import TradeSuggestionsService
from app.core.domain.entities.stock import Stock
from app.core.domain.models.trade_suggestion import DetailedTradeSuggestion
from app.core.repositories.stock_repository import StockRepository
from app.core.ai.tradewise_ai import TradewiseAI
from typing import List
from decimal import Decimal
import pandas as pd

class TradeSuggestionsUseCase:
    def __init__(self, stock_repository: StockRepository):
        self.stock_repository = stock_repository
        self.tradewise_ai = TradewiseAI()

    async def generate_suggestions(self) -> List[DetailedTradeSuggestion]:
        stocks = await self.stock_repository.get_all_stocks()
        df = pd.DataFrame([stock.__dict__ for stock in stocks])
        suggestions = self.tradewise_ai.generate_trade_suggestions(df)
        return [DetailedTradeSuggestion(**suggestion) for suggestion in suggestions]

    async def generate_suggestion_for_stock(self, symbol: str) -> DetailedTradeSuggestion:
        stock = await self.stock_repository.get_stock_by_symbol(symbol)
        if not stock:
            raise ValueError(f"Stock with symbol {symbol} not found")
        df = pd.DataFrame([stock.__dict__])
        suggestions = self.tradewise_ai.generate_trade_suggestions(df)
        return DetailedTradeSuggestion(**suggestions[0]) if suggestions else None

    async def get_top_suggestions(self, limit: int = 5) -> List[DetailedTradeSuggestion]:
        suggestions = await self.generate_suggestions()
        return sorted(suggestions, key=lambda x: x.confidence, reverse=True)[:limit]

    async def get_suggestions_by_action(self, action: str) -> List[DetailedTradeSuggestion]:
        suggestions = await self.generate_suggestions()
        return [suggestion for suggestion in suggestions if suggestion.action.lower() == action.lower()]

# Example usage:
# async def main():
#     stock_repository = StockRepository()
#     trade_suggestions_service = SimpleTradeSuggestionsService()
#     use_case = TradeSuggestionsUseCase(stock_repository, trade_suggestions_service)
#
#     suggestions = await use_case.generate_suggestions()
#     top_suggestions = await use_case.get_top_suggestions(limit=3)
#     buy_suggestions = await use_case.get_suggestions_by_action("BUY")
#
#     print("All suggestions:", suggestions)
#     print("Top 3 suggestions:", top_suggestions)
#     print("Buy suggestions:", buy_suggestions)
#
# if __name__ == "__main__":
#     import asyncio
#     asyncio.run(main())
