from app.core.domain.services.trade_suggestions_service_interface import TradeSuggestionsServiceInterface

class TradeSuggestions:
    def __init__(self, service: TradeSuggestionsServiceInterface):
        self.service = service

    async def generate_suggestions(self):
        return await self.service.suggest_trades()
