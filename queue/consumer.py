import pika
import time

def create_connection():
    while True:
        try:
            return pika.BlockingConnection(pika.ConnectionParameters('localhost'))
        except pika.exceptions.AMQPConnectionError:
            print("Connection failed, retrying in 5 seconds...")
            time.sleep(5)

def callback(ch, method, properties, body):
    print(f"Received {body}")

def start_consuming():
    connection = create_connection()
    channel = connection.channel()
    channel.queue_declare(queue='task_queue', durable=True)
    channel.basic_consume(queue='task_queue', on_message_callback=callback, auto_ack=True)
    print('Waiting for messages. To exit press CTRL+C')
    channel.start_consuming()

if __name__ == "__main__":
    start_consuming()
