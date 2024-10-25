from kafka import KafkaConsumer
from app.core.use_cases.fetch_and_ingest import ingest_data_to_postgres

def start_consuming():
    consumer = KafkaConsumer(
        'task_queue',
        bootstrap_servers='localhost:9092',
        auto_offset_reset='earliest',
        enable_auto_commit=True,
        group_id='my-group'
    )
    print('Waiting for messages. To exit press CTRL+C')
    for message in consumer:
        data = message.value.decode('utf-8')
        ingest_data_to_postgres(data, 'your_table_name')

if __name__ == "__main__":
    start_consuming()
