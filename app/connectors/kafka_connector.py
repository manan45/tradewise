import json
import pandas as pd
from kafka import KafkaProducer, KafkaConsumer
from typing import Callable, Any

class KafkaConnector:
    def __init__(self, bootstrap_servers: str = 'localhost:9092'):
        self.bootstrap_servers = bootstrap_servers
        self.producer = None
        self.consumer = None

    def create_producer(self) -> None:
        if not self.producer:
            self.producer = KafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8')
            )

    def create_consumer(self, topic: str, group_id: str) -> None:
        if not self.consumer:
            self.consumer = KafkaConsumer(
                topic,
                bootstrap_servers=self.bootstrap_servers,
                auto_offset_reset='earliest',
                enable_auto_commit=True,
                group_id=group_id,
                value_deserializer=lambda x: json.loads(x.decode('utf-8'))
            )

    def produce(self, topic: str, message: Any) -> None:
        if not self.producer:
            self.create_producer()
        self.producer.send(topic, message)
        self.producer.flush()

    def produce_dataframe(self, topic: str, dataframe: pd.DataFrame) -> None:
        for _, row in dataframe.iterrows():
            self.produce(topic, row.to_dict())

    def consume(self, topic: str, group_id: str, process_message: Callable[[Any], None]) -> None:
        self.create_consumer(topic, group_id)
        for message in self.consumer:
            process_message(message.value)

    def consume_and_process(self, topic: str, group_id: str, callback: Callable[[pd.DataFrame], None]) -> None:
        def process_message(message):
            dataframe = pd.DataFrame([message])
            callback(dataframe)
        
        self.consume(topic, group_id, process_message)

    def close(self) -> None:
        if self.producer:
            self.producer.close()
        if self.consumer:
            self.consumer.close()
