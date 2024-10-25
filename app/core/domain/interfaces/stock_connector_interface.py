from abc import ABC, abstractmethod
import pandas as pd
from typing import List, Dict, Any

class StockConnectorInterface(ABC):
    @abstractmethod
    def fetch_historical_data(self, security_id: str, exchange_segment: str, from_date: str, to_date: str) -> pd.DataFrame:
        pass

    @abstractmethod
    def fetch_real_time_data(self, security_ids: List[str], exchange_segment: str) -> pd.DataFrame:
        pass

    @abstractmethod
    def get_latest_stock_data(self, security_ids: List[str], exchange_segment: str) -> pd.DataFrame:
        pass

    @abstractmethod
    def get_company_info(self, security_id: str) -> Dict[str, Any]:
        pass
