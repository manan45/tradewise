from kafka import KafkaProducer

def create_producer():
    return KafkaProducer(bootstrap_servers='localhost:9092')

def send_message(message):
    producer = create_producer()
    producer.send('task_queue', value=message.encode('utf-8'))
    producer.flush()
    print(f"Sent {message}")
    producer.close()

if __name__ == "__main__":
    send_message('Hello World!')
