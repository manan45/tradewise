from kafka import KafkaProducer
import time

def create_producer():
    while True:
        try:
            return KafkaProducer(bootstrap_servers='localhost:9092')
        except Exception as e:
            print(f"Connection failed, retrying in 5 seconds... Error: {e}")
            time.sleep(5)

def send_message(message):
    producer = create_producer()
    producer.send('task_queue', value=message.encode('utf-8'))
    print(f"Sent {message}")
    producer.close()

if __name__ == "__main__":
    send_message('Hello World!')
from kafka import KafkaProducer

def create_producer():
    return KafkaProducer(bootstrap_servers='localhost:9092')

def send_message(producer, topic, message):
    producer.send(topic, value=message.encode('utf-8'))
    producer.flush()
