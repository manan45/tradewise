# Queue consumer implementation

class QueueConsumer:
    def __init__(self, queue_name):
        self.queue_name = queue_name

    def consume(self, callback):
        # Implement message consumption logic and call the callback with the message
        pass
