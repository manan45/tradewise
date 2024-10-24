from app.core.domain.entities import Stock, TradeSuggestion

class TradeSuggestions:
    def __init__(self, stock_repository):
        self.stock_repository = stock_repository

    def get_suggestions(self):
        stocks = self.stock_repository.get_all_stocks()
        return [TradeSuggestion(
            action="SELL" if stock.price > 100 else "BUY",
            price=stock.price,
            confidence=0.5,  # Placeholder for actual logic
            stop_loss=stock.price * 0.95,
            order_limit=stock.price * 1.05,
            max_risk=stock.price * 0.05,
            max_reward=stock.price * 0.1
        ) for stock in stocks]
