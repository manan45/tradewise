from app.core.drivers.dhanhq.dhanhq import dhanhq
from app.pipelines.fetcher import QueueProducer
from app.pipelines.ingestor import QueueConsumer
from app.core.use_cases.fetch_and_ingest import aggregate_and_ingest
import pandas as pd
from datetime import datetime, timedelta

def fetch_historical_data(dhan_api, security_id, exchange_segment, from_date, to_date):
    historical_data = dhan_api.historical_daily_data(
        security_id=security_id,
        exchange_segment=exchange_segment,
        instrument_type='EQUITY',
        from_date=from_date,
        to_date=to_date
    )
    
    if historical_data['status'] == 'success':
        df = pd.DataFrame(historical_data['data'])
        df['timestamp'] = pd.to_datetime(df['date'])
        df = df.drop('date', axis=1)
        return df
    else:
        raise Exception(f"Failed to fetch historical data: {historical_data['remarks']}")

def main():
    dhan_api = dhanhq(client_id='your_client_id', access_token='your_access_token')
    producer = QueueProducer(queue_name='stock_data')
    consumer = QueueConsumer(queue_name='stock_data')

    # Example: Fetch historical data for HDFC Bank
    security_id = '1333'  # HDFC Bank
    exchange_segment = dhan_api.NSE
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')

    historical_data = fetch_historical_data(dhan_api, security_id, exchange_segment, start_date, end_date)

    producer.produce_dataframe(historical_data)

    consumer.consume_and_process(lambda df: aggregate_and_ingest(df, 'stock_data'))

if __name__ == "__main__":
    main()



import requests
import pandas as pd
from datetime import datetime

def fetch_data_from_api(api_url):
    """
    Fetch data from a given API URL.

    :param api_url: The API endpoint URL.
    :return: DataFrame containing the fetched data.
    """
    response = requests.get(api_url)
    data = response.json()
    df = pd.DataFrame(data)
    return df

def fetch_real_time_data():
    """
    Fetch real-time data from multiple sources.

    :return: Combined DataFrame of all data sources.
    """
    # Example API URLs
    news_api_url = "https://api.example.com/news"
    oi_api_url = "https://api.example.com/open_interest"
    
    news_data = fetch_data_from_api(news_api_url)
    oi_data = fetch_data_from_api(oi_api_url)
    
    # Combine data
    combined_data = pd.concat([news_data, oi_data], axis=1)
    combined_data['timestamp'] = datetime.now()
    return combined_data
# Data ingestion utilities

def ingest_data(source):
    # Implement data ingestion logic
    pass



import pandas as pd
import os

def load_stock_data() -> pd.DataFrame:
    """
    Load sample stock data for testing and development.

    :return: DataFrame of stock data.
    """
    # Create a sample dataset
    data = {
        'date': pd.date_range(start='2023-01-01', end='2023-12-31', freq='D'),
        'open': [100 + i * 0.1 for i in range(365)],
        'high': [101 + i * 0.1 for i in range(365)],
        'low': [99 + i * 0.1 for i in range(365)],
        'close': [100.5 + i * 0.1 for i in range(365)],
        'volume': [1000000 + i * 1000 for i in range(365)]
    }
    
    df = pd.DataFrame(data)
    
    # Add some random news sentiment
    df['news_sentiment'] = pd.Series([-1, 0, 1]).sample(n=365, replace=True).values
    
    return df

def get_latest_stock_data() -> pd.DataFrame:
    """
    Get the latest stock data for real-time processing.

    :return: DataFrame of the latest stock data.
    """
    df = load_stock_data()
    return df.tail(1)
# Data loading utilities

def load_data():
    # Implement data loading logic
    pass
