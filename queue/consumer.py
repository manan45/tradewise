import pika

def create_connection():
    return pika.BlockingConnection(pika.ConnectionParameters('localhost'))

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
