import asyncio
import websockets
from datetime import datetime, timedelta
import pandas as pd
from dhanhq import dhanhq
from tenacity import retry, wait_fixed, stop_after_attempt
from app.mongodb_client import MongoDBClient

# Initialize the DhanHQ client
client = dhanhq(
    client_id="tradewise",
    access_token="eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzUxMiJ9.eyJpc3MiOiJkaGFuIiwicGFydG5lcklkIjoiIiwiZXhwIjoxNzMyMjE5NjE1LCJ0b2tlbkNvbnN1bWVyVHlwZSI6IlNFTEYiLCJ3ZWJob29rVXJsIjoiIiwiZGhhbkNsaWVudElkIjoiMTEwMTIyOTY2NSJ9.9GDCcMbrzJc9cdKKDGvj_hQJXpqayb7rgh4srpY5gTL9QTE1LfnvsCBTve85kNNw3DXr-33UY8diDy4090_UEQ"
)

@retry(wait=wait_fixed(2), stop=stop_after_attempt(5))
def fetch_user_trade_history():
    """Fetch user's trade history"""
    # Check if the 'get_trades' method exists
    if not hasattr(client, 'get_trades'):
        print("Error: 'get_trades' method not found. Please check the dhanhq library documentation.")
        return None

    trades = client.get_trades()
    if isinstance(trades, dict) and 'data' in trades:
        return pd.DataFrame(trades['data'])
    else:
        print("Unexpected format for trade history")
        return None

@retry(wait=wait_fixed(2), stop=stop_after_attempt(5))
def fetch_stock_data(symbol, interval):
    """Fetch stock data for a given symbol and interval"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)  # Fetch data for the last 30 days
    
    data = client.intraday_minute_data(
        security_id=symbol,
        exchange_segment=client.NSE,
        instrument_type=client.EQ
    )
    
    if not isinstance(data, dict):
        print(f"Unexpected data format received for {symbol}: {type(data)}")
        return None

    # Check if 'data' key exists and is not None
    if 'data' not in data or data['data'] is None:
        print(f"No data received for {symbol}")
        return None

    # Handle both list and dict formats
    if isinstance(data['data'], list):
        df = pd.DataFrame(data['data'])
    elif isinstance(data['data'], dict):
        df = pd.DataFrame([data['data']])
    else:
        print(f"Unexpected data structure for {symbol}: {type(data['data'])}")
        return None

    if df.empty:
        print(f"No data received for {symbol}")
        return None

    # Ensure all required columns are present
    required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required_columns):
        print(f"Missing required columns for {symbol}. Available columns: {df.columns.tolist()}")
        return None

    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.set_index('timestamp')
    df = df.sort_index()
    
    # Convert numeric columns to appropriate types
    numeric_columns = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Resample data to the desired interval
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
    
    if df is not None:
        mongo_client = MongoDBClient(uri="mongodb://mongo:27017")
        mongo_client.insert_stock_data(df.to_dict())
    return df.dropna()

def fetch_all_intervals(symbol):
    """Fetch stock data for all specified intervals"""
    intervals = ["5minute", "15minute", "30minute", "1hour", "1day", "1week"]
    data = {}
    for interval in intervals:
        if interval == "1week":
            # Use a different method for weekly data
            interval_data = fetch_weekly_data(symbol)
        else:
            interval_data = fetch_stock_data(symbol, interval)
        
        if interval_data is not None:
            data[interval] = interval_data
        else:
            print(f"Failed to fetch data for {symbol} at {interval} interval")
        time.sleep(1)  # Add a small delay to avoid hitting rate limits
    return data

def fetch_weekly_data(symbol):
    """Fetch weekly data for a given symbol"""
    try:
        # Implement the logic to fetch weekly data here
        # This might involve using a different API method or
        # aggregating daily data to weekly
        pass
    except Exception as e:
        print(f"Error fetching weekly data for {symbol}: {str(e)}")
        return None

async def main():
    uri = "ws://localhost:8000/ws"
    async with websockets.connect(uri) as websocket:
        while True:
            # Simulate fetching data
            symbol = "RELIANCE"
            stock_data = fetch_all_intervals(symbol)
            await websocket.send(f"Data for {symbol}: {stock_data}")
            await asyncio.sleep(5)  # Simulate real-time interval
    # Fetch user's trade history
    # trade_history = fetch_user_trade_history()
    # if trade_history is not None:
    #     print("User Trade History:")
    #     print(trade_history)
    
    # Fetch stock data for a specific symbol (e.g., RELIANCE)
    symbol = "RELIANCE"
    stock_data = fetch_all_intervals(symbol)
    
    for interval, data in stock_data.items():
        if data is not None:
            print(f"\nStock Data for {symbol} at {interval} interval:")
            print(data.head())

if __name__ == "__main__":
    main()
import requests

class DhanAPI:
    def __init__(self, api_key):
        self.api_key = api_key

    def fetch_stock_data(self, symbol):
        # Implement API call to fetch stock data
        response = requests.get(f"https://api.dhan.com/stocks/{symbol}", headers={"Authorization": f"Bearer {self.api_key}"})
        return response.json()
