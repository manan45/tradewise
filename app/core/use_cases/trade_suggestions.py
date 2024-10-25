from app.core.domain.services.trade_suggestions_service import TradeSuggestionsService

class TradeSuggestions:
    def __init__(self, stock_repository):
        self.service = TradeSuggestionsService(stock_repository)

    async def generate_suggestions(self):
        return await self.service.suggest_trades()
