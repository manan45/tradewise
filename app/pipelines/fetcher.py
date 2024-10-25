import pandas as pd
from app.connectors.kafka_connector import KafkaConnector

class QueueProducer:
    def __init__(self, topic: str):
        self.topic = topic
        self.kafka_connector = KafkaConnector()

    def produce(self, message):
        self.kafka_connector.produce(self.topic, message)
        print(f"Sent message to {self.topic}")

    def produce_dataframe(self, dataframe: pd.DataFrame):
        self.kafka_connector.produce_dataframe(self.topic, dataframe)

    def close(self):
        self.kafka_connector.close()

def main():
    producer = QueueProducer('stock_data')
    try:
        # Example usage
        producer.produce('Hello World!')
        
        # Example with DataFrame
        df = pd.DataFrame({'col1': [1, 2], 'col2': ['a', 'b']})
        producer.produce_dataframe(df)
    finally:
        producer.close()

if __name__ == "__main__":
    main()
