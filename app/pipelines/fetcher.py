import json
import pandas as pd
from kafka import KafkaProducer

class QueueProducer:
    def __init__(self, queue_name):
        self.queue_name = queue_name
        self.producer = KafkaProducer(bootstrap_servers='localhost:9092')

    def produce(self, message):
        self.producer.send(self.queue_name, value=message.encode('utf-8'))
        self.producer.flush()
        print(f"Sent message to {self.queue_name}")

    def produce_dataframe(self, dataframe):
        for index, row in dataframe.iterrows():
            message = row.to_json()
            self.produce(message)

    def close(self):
        self.producer.close()

if __name__ == "__main__":
    producer = QueueProducer('task_queue')
    try:
        # Example usage
        producer.produce('Hello World!')
        
        # Example with DataFrame
        df = pd.DataFrame({'col1': [1, 2], 'col2': ['a', 'b']})
        producer.produce_dataframe(df)
    finally:
        producer.close()
