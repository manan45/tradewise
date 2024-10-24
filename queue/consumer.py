from kafka import KafkaConsumer

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
        print(f"Received {message.value.decode('utf-8')}")

if __name__ == "__main__":
    start_consuming()
