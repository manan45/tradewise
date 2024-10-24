import asyncio
import pandas as pd
from dhanhq import dhanhq
import requests
from tenacity import retry, wait_fixed, stop_after_attempt
from app.core.drivers.mongodb_client import get_database

# Initialize the DhanHQ client
client = dhanhq(
    client_id="tradewise",
    access_token="your_access_token"
)

@retry(wait=wait_fixed(2), stop=stop_after_attempt(5))
async def fetch_stock_data(symbol, interval):
    """Fetch stock data for a given symbol and interval asynchronously."""
    data = client.intraday_minute_data(
        security_id=symbol,
        exchange_segment=client.NSE,
        instrument_type=client.EQ
    )
    
    if not isinstance(data, dict) or 'data' not in data or data['data'] is None:
        print(f"No data received for {symbol}")
        return None

    df = pd.DataFrame(data['data'] if isinstance(data['data'], list) else [data['data']])

    if df.empty:
        print(f"No data received for {symbol}")
        return None

    required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required_columns):
        print(f"Missing required columns for {symbol}. Available columns: {df.columns.tolist()}")
        return None

    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.set_index('timestamp').sort_index()
    
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    resampling_dict = {
        '5minute': '5T', '15minute': '15T', '30minute': '30T',
        '1hour': '1H', '1day': '1D', '1week': '1W'
    }
    if interval in resampling_dict:
        df = df.resample(resampling_dict[interval]).agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
        })
    else:
        print(f"Invalid interval: {interval}")
        return None
    
    if not df.empty:
        db = get_database('stock_data')
        collection = db[f'{symbol}_{interval}']
        collection.insert_many(df.reset_index().to_dict('records'))
    
    return df.dropna()

async def fetch_all_intervals(symbol):
    """Fetch stock data for all specified intervals asynchronously."""
    intervals = ["5minute", "15minute", "30minute", "1hour", "1day", "1week"]
    tasks = [fetch_stock_data(symbol, interval) for interval in intervals]
    results = await asyncio.gather(*tasks)
    return {interval: result for interval, result in zip(intervals, results) if result is not None}

async def main():
    symbol = "RELIANCE"
    stock_data = await fetch_all_intervals(symbol)
    print(f"Data for {symbol}: {stock_data}")

if __name__ == "__main__":
    asyncio.run(main())

class DhanAPI:
    def __init__(self, api_key):
        self.api_key = api_key

    def fetch_stock_data(self, symbol):
        response = requests.get(f"https://api.dhan.com/stocks/{symbol}", headers={"Authorization": f"Bearer {self.api_key}"})
        return response.json()
