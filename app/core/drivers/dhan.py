import pandas as pd
from dhanhq import dhanhq
from tenacity import retry, wait_fixed, stop_after_attempt
from websocket import WebSocketApp, enableTrace
import json
import threading
import os
import logging

class DhanAPI:
    def __init__(self):
        self.client = dhanhq(
            client_id=os.getenv("DHAN_CLIENT_ID", "tradewise"),
            access_token=os.getenv("DHAN_ACCESS_TOKEN", "your_access_token")
        )
        self.ws = None
        self.thread = None
        logging.basicConfig(level=logging.INFO)

    @retry(wait=wait_fixed(2), stop=stop_after_attempt(5))
    def fetch_stock_data(self, symbol, interval):
        """Fetch stock data for a given symbol and interval."""
        data = self.client.intraday_minute_data(
            security_id=symbol,
            exchange_segment=self.client.NSE,
            instrument_type=self.client.EQ
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
        
        return df.dropna()

    def fetch_all_intervals(self, symbol):
        """Fetch stock data for all specified intervals."""
        intervals = ["5minute", "15minute", "30minute", "1hour", "1day", "1week"]
        results = {}
        for interval in intervals:
            result = self.fetch_stock_data(symbol, interval)
            if result is not None:
                results[interval] = result
        return results

    def on_message(self, ws, message):
        data = json.loads(message)
        logging.info(f"Received data: {data}")
        # TODO: Process the data as needed

    def on_error(self, ws, error):
        logging.error(f"Error: {error}")

    def on_close(self, ws, close_status_code, close_msg):
        logging.info("WebSocket connection closed")

    def on_open(self, ws):
        logging.info("WebSocket connection opened")

    def connect_websocket(self):
        enableTrace(True)
        self.ws = WebSocketApp(
            f"wss://api-feed.dhan.co?version=2&token={self.client.access_token}&clientId={self.client.client_id}&authType=2",
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close,
            on_open=self.on_open
        )
        
        self.thread = threading.Thread(target=self._run_websocket)
        self.thread.daemon = True
        self.thread.start()

    def _run_websocket(self):
        self.ws.run_forever()

    def disconnect_websocket(self):
        if self.ws:
            self.ws.close()
            self.ws = None
        if self.thread:
            self.thread.join()
            self.thread = None

    def subscribe_symbols(self, symbols):
        if self.ws and self.ws.sock and self.ws.sock.connected:
            try:
                subscribe_message = {
                    "action": "subscribe",
                    "params": {
                        "symbols": symbols
                    }
                }
                self.ws.send(json.dumps(subscribe_message))
                logging.info(f"Subscribed to symbols: {symbols}")
            except Exception as e:
                logging.error(f"Failed to subscribe to symbols: {e}")
        else:
            logging.warning("WebSocket is not connected. Cannot subscribe to symbols.")

# Usage example
dhan_api = DhanAPI()

def main():
    symbol = "RELIANCE"
    stock_data = dhan_api.fetch_all_intervals(symbol)
    print(f"Data for {symbol}: {stock_data}")

    dhan_api.connect_websocket()
    try:
        dhan_api.subscribe_symbols(["RELIANCE", "INFY", "TCS"])
        # Keep the main thread running
        while True:
            pass
    except KeyboardInterrupt:
        print("Interrupted by user, closing connection.")
    finally:
        dhan_api.disconnect_websocket()

if __name__ == "__main__":
    main()
