import pika
import time

def create_connection():
    while True:
        try:
            return pika.BlockingConnection(pika.ConnectionParameters('localhost'))
        except pika.exceptions.AMQPConnectionError:
            print("Connection failed, retrying in 5 seconds...")
            time.sleep(5)

def send_message(message):
    connection = create_connection()
    channel = connection.channel()
    channel.queue_declare(queue='task_queue', durable=True)
    channel.basic_publish(exchange='',
                          routing_key='task_queue',
                          body=message,
                          properties=pika.BasicProperties(
                             delivery_mode=2,  # make message persistent
                          ))
    print(f"Sent {message}")
    connection.close()

if __name__ == "__main__":
    send_message('Hello World!')
