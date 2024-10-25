from app.connectors.dhanhq.dhanhq import dhanhq
from app.pipelines.fetcher import QueueProducer
from app.pipelines.ingestor import QueueConsumer
from app.core.use_cases.fetch_and_ingest import FetchAndIngestUseCase
from app.core.repositories.stock_repository import StockRepository
import pandas as pd
from datetime import datetime, timedelta
import requests
from typing import List
import os
from dotenv import load_dotenv

class DataService:
    def __init__(self, client_id: str, access_token: str):
        self.dhan_api = dhanhq(client_id=client_id, access_token=access_token)
        self.producer = QueueProducer(topic='stock_data')
        self.consumer = QueueConsumer(topic='stock_data', group_id='data_service_group')
        self.stock_repository = StockRepository()
        self.fetch_and_ingest_use_case = FetchAndIngestUseCase(self.stock_repository)

    def fetch_historical_data(self, security_id: str, exchange_segment: str, from_date: str, to_date: str) -> pd.DataFrame:
        historical_data = self.dhan_api.historical_daily_data(
            security_id=security_id,
            exchange_segment=exchange_segment,
            instrument_type='EQUITY',
            from_date=from_date,
            to_date=to_date
        )
        
        if historical_data['status'] == 'success':
            df = pd.DataFrame(historical_data['data'])
            if 'date' in df.columns:
                df['timestamp'] = pd.to_datetime(df['date'])
                df = df.drop('date', axis=1)
            elif 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            else:
                raise KeyError("Neither 'date' nor 'timestamp' column found in the data")
            return df
        else:
            raise Exception(f"Failed to fetch historical data: {historical_data['remarks']}")

    def fetch_real_time_data(self, api_url: str) -> pd.DataFrame:
        response = requests.get(api_url)
        data = response.json()
        df = pd.DataFrame(data)
        df['timestamp'] = datetime.now()
        return df

    def fetch_and_process_data(self, security_ids: List[str], exchange_segment: str):
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')

        for security_id in security_ids:
            historical_data = self.fetch_historical_data(security_id, exchange_segment, start_date, end_date)
            self.producer.produce_dataframe(historical_data)

        self.consumer.consume_and_process(self.fetch_and_ingest_use_case.fetch_and_ingest_stock_data)

    def load_sample_data(self) -> pd.DataFrame:
        data = {
            'timestamp': pd.date_range(start='2023-01-01', end='2023-12-31', freq='D'),
            'open': [100 + i * 0.1 for i in range(365)],
            'high': [101 + i * 0.1 for i in range(365)],
            'low': [99 + i * 0.1 for i in range(365)],
            'close': [100.5 + i * 0.1 for i in range(365)],
            'volume': [1000000 + i * 1000 for i in range(365)]
        }
        
        df = pd.DataFrame(data)
        df['news_sentiment'] = pd.Series([-1, 0, 1]).sample(n=365, replace=True).values
        return df

    def get_latest_stock_data(self) -> pd.DataFrame:
        df = self.load_sample_data()
        return df.tail(1)

def main():
    load_dotenv()  # This will load environment variables from a .env file
    client_id = os.getenv('DHAN_CLIENT_ID', 'tradewise')
    access_token = os.getenv('DHAN_ACCESS_TOKEN', 'eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzUxMiJ9.eyJpc3MiOiJkaGFuIiwicGFydG5lcklkIjoiIiwiZXhwIjoxNzMyMjE5NjE1LCJ0b2tlbkNvbnN1bWVyVHlwZSI6IlNFTEYiLCJ3ZWJob29rVXJsIjoiIiwiZGhhbkNsaWVudElkIjoiMTEwMTIyOTY2NSJ9.9GDCcMbrzJc9cdKKDGvj_hQJXpqayb7rgh4srpY5gTL9QTE1LfnvsCBTve85kNNw3DXr-33UY8diDy4090_UEQ')
    
    if not client_id or not access_token:
        raise ValueError("DHAN_CLIENT_ID and DHAN_ACCESS_TOKEN must be set in the environment or .env file")
    
    data_service = DataService(client_id=client_id, access_token=access_token)
    security_ids = ['1333']  # HDFC Bank
    exchange_segment = data_service.dhan_api.NSE
    data_service.fetch_and_process_data(security_ids, exchange_segment)

if __name__ == "__main__":
    main()
