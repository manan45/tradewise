import os
import logging
from dotenv import load_dotenv
from app.connectors.stock_app_apple import AppleStocksConnector
from app.services.base_data_service import BaseDataService
from app.connectors.kafka_connector import KafkaConnector
import pandas as pd
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AppleDataService(BaseDataService):
    def __init__(self):
        super().__init__()
        self.stock_api = AppleStocksConnector()
        self.kafka_connector = KafkaConnector()

    def fetch_historical_data(self, symbol: str, interval: str = '1d', range: str = '1mo') -> pd.DataFrame:
        try:
            logger.info(f"Fetching historical data for {symbol}")
            df = self.stock_api.get_stock_data(symbol, interval, range)
            if df.empty:
                logger.warning(f"No historical data available for {symbol}")
                return pd.DataFrame()
            df.reset_index(inplace=True)
            df.rename(columns={'index': 'timestamp'}, inplace=True)
            logger.info(f"Successfully fetched historical data for {symbol}")
            return df
        except Exception as e:
            logger.error(f"Failed to fetch historical data for {symbol}: {str(e)}")
            return pd.DataFrame()

    def fetch_real_time_quote(self, symbol: str) -> pd.DataFrame:
        try:
            logger.info(f"Fetching real-time quote for {symbol}")
            quote = self.stock_api.get_real_time_quote(symbol)
            if not quote:
                logger.warning(f"No real-time quote available for {symbol}")
                return pd.DataFrame()
            df = pd.DataFrame([quote])
            logger.info(f"Successfully fetched real-time quote for {symbol}")
            return df
        except Exception as e:
            logger.error(f"Failed to fetch real-time quote for {symbol}: {str(e)}")
            raise

class DhanDataService(BaseDataService):
    def __init__(self):
        super().__init__()
        self.client_id = os.getenv('DHAN_CLIENT_ID')
        self.access_token = os.getenv('DHAN_ACCESS_TOKEN')
        if not self.client_id or not self.access_token:
            raise ValueError("DHAN_CLIENT_ID and DHAN_ACCESS_TOKEN must be set in the environment or .env file")
        self.dhan_api = dhanhq(client_id=self.client_id, access_token=self.access_token)
        self.kafka_connector = KafkaConnector()

    def fetch_historical_data(self, security_id: str, exchange_segment: str, from_date: str, to_date: str) -> pd.DataFrame:
        try:
            historical_data = self.dhan_api.historical_daily_data(
                security_id=security_id,
                exchange_segment=exchange_segment,
                instrument_type='EQUITY',
                from_date=from_date,
                to_date=to_date
            )
            
            if historical_data['status'] == 'success':
                df = pd.DataFrame(historical_data['data'])
                df['timestamp'] = pd.to_datetime(df['date'])
                df = df.drop('date', axis=1)
                return df
            else:
                raise Exception(f"Failed to fetch historical data: {historical_data['remarks']}")
        except Exception as e:
            logger.error(f"Failed to fetch historical data for {security_id}: {str(e)}")
            raise

    def fetch_real_time_quote(self, security_id: str) -> pd.DataFrame:
        try:
            quote = self.dhan_api.get_quote(security_id)
            if quote['status'] == 'success':
                df = pd.DataFrame([quote['data']])
                df['timestamp'] = datetime.now()
                return df
            else:
                raise Exception(f"Failed to fetch real-time quote: {quote['remarks']}")
        except Exception as e:
            logger.error(f"Failed to fetch real-time quote for {security_id}: {str(e)}")
            return pd.DataFrame()

def main():
    load_dotenv()
    data_service = AppleDataService()
    symbols = ['AAPL']  # Example symbols, including DHAN
    
    try:
        # if data_source == 'apple':
        #     data_service = AppleDataService()
        #     symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'FB']  # Example symbols
        # elif data_source == 'dhan':
        #     data_service = DhanDataService()
        #     symbols = ['1333']  # Example security_id for HDFC Bank
        # else:
        #     raise ValueError(f"Invalid DATA_SOURCE: {data_source}. Must be 'apple' or 'dhan'.")
        
        data_service = AppleDataService()
        symbols = ['AAPL']  # Example symbols
        data_service.fetch_and_process_data(symbols)
    except Exception as e:
        logger.error(f"An error occurred in main: {str(e)}")

if __name__ == "__main__":
    main()
