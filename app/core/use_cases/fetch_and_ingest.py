from sqlalchemy import create_engine
import pandas as pd


def ingest_data_to_postgres(data, table_name):
    engine = 
    data.to_sql(table_name, engine, if_exists='append', index=True)

def aggregate_and_ingest(dataframe, table_name):
    timeframes = ['1min', '2min', '5min', '15min', '30min', '1H', '2H', '4H', 'D', 'W', 'M', 'Y']
    
    if 'timestamp' not in dataframe.columns:
        raise ValueError("Dataframe must contain a 'timestamp' column")
    
    dataframe['timestamp'] = pd.to_datetime(dataframe['timestamp'])
    dataframe.set_index('timestamp', inplace=True)
    
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in dataframe.columns for col in required_columns):
        raise ValueError(f"Dataframe must contain all of these columns: {required_columns}")
    
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

# Example usage:
# df = pd.DataFrame({
#     'timestamp': pd.date_range(start='2023-01-01', end='2023-01-02', freq='1min'),
#     'open': [100] * 1440,
#     'high': [110] * 1440,
#     'low': [90] * 1440,
#     'close': [105] * 1440,
#     'volume': [1000] * 1440
# })
# aggregate_and_ingest(df, 'stock_data')
