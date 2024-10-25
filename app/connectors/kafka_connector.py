import json
import pandas as pd
from typing import Callable, Any
import logging
import os
from kafka import KafkaProducer, KafkaConsumer
from kafka.errors import NoBrokersAvailable, KafkaError
import logging.config
from datetime import datetime

class KafkaConnector:
    def __init__(self):
        self.bootstrap_servers = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:29092')
        self.producer = None
        self.consumer = None
        self.logger = logging.getLogger(__name__)
        
        # Configure logging for Kafka
        logging.config.dictConfig({
            'version': 1,
            'disable_existing_loggers': False,
            'loggers': {
                'kafka': {
                    'level': 'WARNING',  # Change this to ERROR if you want even fewer logs
                },
            },
        })

    def create_producer(self) -> None:
        if not self.producer:
            try:
                self.producer = KafkaProducer(
                    bootstrap_servers=self.bootstrap_servers,
                    value_serializer=lambda v: json.dumps(v, default=self.json_serializer).encode('utf-8')
                )
                self.logger.info(f"Successfully connected to Kafka broker at {self.bootstrap_servers}")
            except NoBrokersAvailable:
                self.logger.error(f"Unable to connect to Kafka broker at {self.bootstrap_servers}. Please check if Kafka is running and the address is correct.")
                raise

    @staticmethod
    def json_serializer(obj):
        """JSON serializer for objects not serializable by default json code"""
        if isinstance(obj, (datetime, pd.Timestamp)):
            return obj.isoformat()
        raise TypeError(f"Type {type(obj)} not serializable")

    def create_consumer(self, topic: str, group_id: str) -> None:
        if not self.consumer:
            try:
                self.consumer = KafkaConsumer(
                    topic,
                    bootstrap_servers=self.bootstrap_servers,
                    group_id=group_id,
                    auto_offset_reset='earliest',
                    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
                )
                self.logger.info(f"Successfully connected to Kafka broker at {self.bootstrap_servers}")
            except KafkaError as e:
                self.logger.error(f"Unable to connect to Kafka broker at {self.bootstrap_servers}. Error: {str(e)}")
                raise

    def produce(self, topic: str, message: Any) -> None:
        if not self.producer:
            self.create_producer()
        try:
            future = self.producer.send(topic, value=message)
            future.get(timeout=10)  # Wait for the message to be sent
            self.producer.flush()
        except Exception as e:
            self.logger.error(f"Error producing message to Kafka: {str(e)}")
            raise

    def produce_dataframe(self, topic: str, dataframe: pd.DataFrame) -> None:
        print(dataframe.columns)
        if 'symbol' not in dataframe.columns:
            self.logger.warning(f"DataFrame does not contain 'symbol' column. Adding dummy symbol.")
            dataframe['symbol'] = 'UNKNOWN'
        
        for _, row in dataframe.iterrows():
            row_dict = row.to_dict()
            # Convert all Timestamp objects to ISO format strings
            for key, value in row_dict.items():
                if isinstance(value, pd.Timestamp):
                    row_dict[key] = value.isoformat()
            self.produce(topic, row_dict)

    def consume(self, topic: str, group_id: str, process_message: Callable[[Any], None]) -> None:
        self.create_consumer(topic, group_id)
        try:
            for message in self.consumer:
                try:
                    value = message.value
                    process_message(value)
                except Exception as e:
                    self.logger.error(f'Error processing message: {str(e)}')
        except KeyboardInterrupt:
            self.logger.info('Aborted by user')
        finally:
            self.consumer.close()

    def consume_and_process(self, topic: str, group_id: str, callback: Callable[[pd.DataFrame], None]) -> None:
        def process_message(message):
            try:
                if isinstance(message, dict) and 'symbol' in message:
                    dataframe = pd.DataFrame([message])
                    callback(dataframe)
                else:
                    self.logger.warning(f"Received message without 'symbol' key: {message}")
            except Exception as e:
                self.logger.error(f"Error processing message: {str(e)}")

        self.consume(topic, group_id, process_message)

    def close(self) -> None:
        if self.producer:
            self.producer.flush()
        if self.consumer:
            self.consumer.close()
