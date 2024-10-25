from kafka import KafkaProducer

def create_producer():
    return KafkaProducer(bootstrap_servers='localhost:9092')

def send_message(data):
    producer = create_producer()
    producer.send('task_queue', value=data.encode('utf-8'))
    producer.flush()
    print(f"Sent {data}")
    producer.close()

if __name__ == "__main__":
    send_message('Hello World!')
