from app.core.infrastructure.drivers.dhan import DhanAPI
from app.interfaces.queue.producer import QueueProducer
from app.interfaces.queue.consumer import QueueConsumer
from app.core.use_cases.fetch_and_ingest import aggregate_and_ingest

def main():
    dhan_api = DhanAPI(api_key='your_api_key')
    producer = QueueProducer(queue_name='stock_data')
    consumer = QueueConsumer(queue_name='stock_data')

    historical_data = dhan_api.fetch_historical_data('AAPL', '2023-01-01', '2023-12-31')

    producer.produce_dataframe(historical_data)

    consumer.consume_and_process(lambda df: aggregate_and_ingest(df, 'stock_data'))

if __name__ == "__main__":
    main()
