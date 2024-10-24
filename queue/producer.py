import pika

def create_connection():
    return pika.BlockingConnection(pika.ConnectionParameters('localhost'))

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
