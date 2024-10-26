import os
import logging
from abc import ABC, abstractmethod
from dotenv import load_dotenv
from app.pipelines.fetcher import QueueProducer
from app.pipelines.ingestor import QueueConsumer
from app.core.use_cases.fetch_and_ingest import FetchAndIngestUseCase
from app.core.repositories.stock_repository import StockRepository
from app.core.repositories.index_repository import IndexRepository
import pandas as pd
from typing import List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseDataService(ABC):
    def __init__(self):
        load_dotenv()
        # self.kafka_bootstrap_servers = os.getenv('KAFKA_BOOTSTRAP_SERVERS')
        # self.producer = QueueProducer(topic='stock_data')
        # self.consumer = QueueConsumer(topic='stock_data', group_id='data_service_group')
        self.stock_repository = StockRepository()
        self.index_repository = IndexRepository()
        self.fetch_and_ingest_use_case = FetchAndIngestUseCase(self.stock_repository)

    # def get_latest_stock_data(self, symbol: str) -> pd.DataFrame:
    #     return self.fetch_real_time_quote(symbol)
