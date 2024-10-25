import pandas as pd
from app.connectors.kafka_connector import KafkaConnector
from app.core.use_cases.fetch_and_ingest import FetchAndIngestUseCase
from app.core.repositories.stock_repository import StockRepository

class QueueConsumer:
    def __init__(self, topic: str, group_id: str):
        self.topic = topic
        self.group_id = group_id
        self.kafka_connector = KafkaConnector()
        self.stock_repository = StockRepository()
        self.fetch_and_ingest_use_case = FetchAndIngestUseCase(self.stock_repository)

    def consume(self, process_message):
        print(f'Waiting for messages on {self.topic}. To exit press CTRL+C')
        self.kafka_connector.consume(self.topic, self.group_id, process_message)

    def consume_and_process(self):
        self.kafka_connector.consume_and_process(self.topic, self.group_id, self.process_data)

    def process_data(self, df: pd.DataFrame):
        symbol = df['symbol'].iloc[0]  # Assuming the dataframe contains a 'symbol' column
        self.fetch_and_ingest_use_case.fetch_and_ingest_stock_data(symbol, df)

def main():
    consumer = QueueConsumer('stock_data', 'stock_ingestor_group')
    consumer.consume_and_process()

if __name__ == "__main__":
    main()
