import pandas as pd
from kafka import KafkaConsumer
from app.core.use_cases.fetch_and_ingest import ingest_data_to_postgres

class QueueConsumer:
    def __init__(self, queue_name):
        self.queue_name = queue_name
        self.consumer = KafkaConsumer(
            self.queue_name,
            bootstrap_servers='localhost:9092',
            auto_offset_reset='earliest',
            enable_auto_commit=True,
            group_id='my-group'
        )

    def consume(self, process_message):
        print(f'Waiting for messages on {self.queue_name}. To exit press CTRL+C')
        for message in self.consumer:
            process_message(message.value)

    def consume_and_process(self, callback):
        def process_message(message):
            dataframe = pd.read_json(message)
            callback(dataframe)
        
        self.consume(process_message)

def start_consuming():
    consumer = QueueConsumer('task_queue')
    consumer.consume_and_process(lambda df: ingest_data_to_postgres(df, 'your_table_name'))

if __name__ == "__main__":
    start_consuming()
