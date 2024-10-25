import pandas as pd
from app.connectors.kafka_connector import KafkaConnector
from app.core.use_cases.fetch_and_ingest import FetchAndIngestUseCase
from app.core.repositories.stock_repository import StockRepository
import logging
import asyncio

logger = logging.getLogger(__name__)

class QueueConsumer:
    def __init__(self, topic: str, group_id: str):
        self.topic = topic
        self.group_id = group_id
        self.kafka_connector = KafkaConnector()
        self.stock_repository = StockRepository()
        self.fetch_and_ingest_use_case = FetchAndIngestUseCase(self.stock_repository)

    def consume(self, process_message):
        logger.info(f'Waiting for messages on {self.topic}. To exit press CTRL+C')
        self.kafka_connector.consume(self.topic, self.group_id, process_message)

    def consume_and_process(self):
        self.kafka_connector.consume_and_process(self.topic, self.group_id, self.process_data)

    def process_data(self, df: pd.DataFrame):
        try:
            if 'symbol' not in df.columns:
                logger.warning("DataFrame does not contain 'symbol' column")
                return
            symbol = df['symbol'].iloc[0]
            asyncio.run(self.fetch_and_ingest_use_case.fetch_and_ingest_stock_data(symbol, df))
        except Exception as e:
            logger.error(f"Error processing data: {str(e)}")

def main():
    logging.basicConfig(level=logging.INFO)
    consumer = QueueConsumer('stock_data', 'stock_ingestor_group')
    try:
        consumer.consume_and_process()
    except KeyboardInterrupt:
        logger.info("Shutting down consumer...")
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
