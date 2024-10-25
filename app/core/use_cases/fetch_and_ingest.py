from sqlalchemy import create_engine
from app.core.drivers.dhan import DhanAPI
import pandas as pd
import asyncio

# Define your PostgreSQL connection string
DATABASE_URI = 'postgresql://user:password@localhost/mydatabase'

def ingest_data_to_postgres(data, table_name):
    engine = create_engine(DATABASE_URI)
    data.to_sql(table_name, engine, if_exists='replace', index=False)

async def fetch_and_ingest_index_data():
    dhan_api = DhanAPI(api_key="your_api_key")
    # Simulate fetching data from Dhan API
    sample_data = {
        "symbol": "NIFTY",
        "channels": {
            "RSI": 70.5,
            "Moving Average": 150.3,
            "Forecast": 155.0
        }
    }
    df = pd.DataFrame([sample_data])
    ingest_data_to_postgres(df, 'index_data')

if __name__ == "__main__":
    asyncio.run(fetch_and_ingest_index_data())
def ingest_data_to_postgres(data, table_name):
    # Logic to store data in Postgres
    pass
# Use case for fetching and ingesting data

class FetchAndIngest:
    def execute(self):
        # Implement fetching and ingestion logic
        pass
import pandas as pd

def ingest_data_to_postgres(data, table_name):
    # Code to ingest data into PostgreSQL
    pass

def aggregate_and_ingest(dataframe, table_name):
    timeframes = ['1min', '2min', '5min', '15min', '30min', '1hour', '2hour', '4hour', 'daily', 'weekly', 'monthly', 'yearly']
    dataframe['timestamp'] = pd.to_datetime(dataframe['timestamp'])
    dataframe.set_index('timestamp', inplace=True)
    aggregated_data = {}
    
    for timeframe in timeframes:
        aggregated_data[timeframe] = dataframe.resample(timeframe).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
    
    for timeframe, data in aggregated_data.items():
        ingest_data_to_postgres(data, f"{table_name}_{timeframe}")
