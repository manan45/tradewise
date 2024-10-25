import os
import logging
from abc import ABC, abstractmethod
from dotenv import load_dotenv
from app.pipelines.fetcher import QueueProducer
from app.pipelines.ingestor import QueueConsumer
from app.core.use_cases.fetch_and_ingest import FetchAndIngestUseCase
from app.core.repositories.stock_repository import StockRepository
import pandas as pd
from typing import List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseDataService(ABC):
    def __init__(self):
        load_dotenv()
        self.kafka_bootstrap_servers = os.getenv('KAFKA_BOOTSTRAP_SERVERS')
        self.producer = QueueProducer(topic='stock_data')
        self.consumer = QueueConsumer(topic='stock_data', group_id='data_service_group')
        self.stock_repository = StockRepository()
        self.fetch_and_ingest_use_case = FetchAndIngestUseCase(self.stock_repository)

    @abstractmethod
    def fetch_historical_data(self, symbol: str, **kwargs) -> pd.DataFrame:
        pass

    @abstractmethod
    def fetch_real_time_quote(self, symbol: str) -> pd.DataFrame:
        pass

    def fetch_and_process_data(self, symbols: List[str]):
        for symbol in symbols:
            try:
                logger.info(f"Processing data for {symbol}")
                historical_data = self.fetch_historical_data(symbol)
                historical_data['symbol'] = symbol
                self.producer.produce_dataframe(historical_data)
                
                real_time_data = self.fetch_real_time_quote(symbol)
                real_time_data['symbol'] = symbol
                self.producer.produce_dataframe(real_time_data)
                
                logger.info(f"Successfully processed data for {symbol}")
            except Exception as e:
                logger.error(f"Error processing data for {symbol}: {str(e)}")

        logger.info("Starting data ingestion process")
        self.consumer.consume_and_process()

    def get_latest_stock_data(self, symbol: str) -> pd.DataFrame:
        return self.fetch_real_time_quote(symbol)
