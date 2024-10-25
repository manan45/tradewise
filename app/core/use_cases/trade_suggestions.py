from app.core.domain.interfaces.trade_suggestions_service import TradeSuggestionsService
from app.core.domain.entities.stock import Stock
from app.core.domain.entities.trade_suggestion import DetailedTradeSuggestion
from app.core.repositories.stock_repository import StockRepository
from typing import List
from decimal import Decimal

class TradeSuggestionsUseCase:
    def __init__(self, stock_repository: StockRepository, trade_suggestions_service: TradeSuggestionsService):
        self.stock_repository = stock_repository
        self.trade_suggestions_service = trade_suggestions_service

    async def generate_suggestions(self) -> List[DetailedTradeSuggestion]:
        stocks = await self.stock_repository.get_all_stocks()
        return await self.trade_suggestions_service.get_trade_suggestions(stocks)

    async def generate_suggestion_for_stock(self, symbol: str) -> DetailedTradeSuggestion:
        stock = await self.stock_repository.get_stock_by_symbol(symbol)
        if not stock:
            raise ValueError(f"Stock with symbol {symbol} not found")
        return await self.trade_suggestions_service.analyze_stock(stock)

    async def get_top_suggestions(self, limit: int = 5) -> List[DetailedTradeSuggestion]:
        suggestions = await self.generate_suggestions()
        return sorted(suggestions, key=lambda x: x.confidence, reverse=True)[:limit]

    async def get_suggestions_by_action(self, action: str) -> List[DetailedTradeSuggestion]:
        suggestions = await self.generate_suggestions()
        return [suggestion for suggestion in suggestions if suggestion.action.lower() == action.lower()]

class SimpleTradeSuggestionsService(TradeSuggestionsService):
    async def get_trade_suggestions(self, stocks: List[Stock]) -> List[DetailedTradeSuggestion]:
        suggestions = []
        for stock in stocks:
            suggestions.append(await self.analyze_stock(stock))
        return suggestions

    async def analyze_stock(self, stock: Stock) -> DetailedTradeSuggestion:
        confidence = await self.calculate_confidence(stock)
        stop_loss = await self.determine_stop_loss(stock)
        order_limit = await self.determine_order_limit(stock)
        max_risk, max_reward = await self.calculate_risk_reward(stock, stop_loss, order_limit)

        latest_price = stock.get_latest_price()
        action = "BUY" if confidence > 0.6 else "SELL" if confidence < 0.4 else "HOLD"

        return DetailedTradeSuggestion(
            action=action,
            price=stock.current_price,
            confidence=confidence,
            stop_loss=stop_loss,
            order_limit=order_limit,
            max_risk=max_risk,
            max_reward=max_reward,
            open=latest_price.open,
            high=latest_price.high,
            low=latest_price.low,
            close=latest_price.close
        )

    async def calculate_confidence(self, stock: Stock) -> float:
        avg_price = stock.calculate_average_price(days=30)
        if stock.current_price > avg_price:
            return min(1.0, (stock.current_price / avg_price) - 0.5)
        else:
            return max(0.0, 1.5 - (avg_price / stock.current_price))

    async def determine_stop_loss(self, stock: Stock) -> Decimal:
        return stock.current_price * Decimal('0.95')

    async def determine_order_limit(self, stock: Stock) -> Decimal:
        return stock.current_price * Decimal('1.05')

    async def calculate_risk_reward(self, stock: Stock, stop_loss: Decimal, order_limit: Decimal) -> tuple[Decimal, Decimal]:
        max_risk = stock.current_price - stop_loss
        max_reward = order_limit - stock.current_price
        return max_risk, max_reward

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
