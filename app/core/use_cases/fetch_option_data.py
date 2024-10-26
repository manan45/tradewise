from app.core.domain.interfaces.stock_repository_interface import StockRepositoryInterface
from app.services.kiteconnect_data_service import KiteConnectDataService
from typing import Dict, Any

class FetchOptionDataUseCase:
    def __init__(self, stock_repository: StockRepositoryInterface):
        self.stock_repository = stock_repository

    async def execute(self, symbol: str, instrument_token: int) -> Dict[str, Any]:
        kite_service = KiteConnectDataService(symbol, instrument_token)
        return await kite_service.fetch_and_process_data()
