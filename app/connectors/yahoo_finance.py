import requests
import pandas as pd
from datetime import datetime, timedelta
import logging
import time

logger = logging.getLogger(__name__)

class YahooFinanceConnector:
    BASE_URL = "https://query1.finance.yahoo.com/v8/finance/chart/"

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        })
        print("YahooFinanceConnector initialized")

    def get_stock_data(self, symbol, interval='1d', start_date=None, end_date=None):
        print(f"Fetching stock data for {symbol} from {start_date} to {end_date} with interval {interval}")
        if not start_date:
            start_date = datetime.now() - timedelta(days=7300)  # Approximately 20 years ago
        if not end_date:
            end_date = datetime.now()

        # Fetch daily data for the entire period
        return self._fetch_data(symbol, interval, start_date, end_date)    
        
    def _fetch_data(self, symbol, interval, start_date, end_date):
        all_data = []
        current_start = start_date
        #  TODO implement a way to fetch data for the last 20 years

        while current_start < end_date:
            current_end = min(current_start + timedelta(days=7), end_date)
            chunk_data = self._fetch_chunk(symbol, interval, current_start, current_end)
            if not chunk_data.empty:
                all_data.append(chunk_data)
            current_start = current_end + timedelta(days=1)
            time.sleep(1)  # To avoid hitting rate limits

        if not all_data:
            print(f"No data available for symbol: {symbol}")
            return pd.DataFrame()

        combined_data = pd.concat(all_data)
        combined_data = combined_data.drop_duplicates().sort_index()
        
        # Handle NaN values in the 'volume' column
        combined_data['volume'] = combined_data['volume'].fillna(0).astype(int)
        
        return combined_data

    def _fetch_chunk(self, symbol, interval, start_date, end_date):
        url = f"{self.BASE_URL}{symbol}"
        params = {
            "interval": interval,
            "period1": int(start_date.timestamp()),
            "period2": int(end_date.timestamp())
        }

        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            if 'chart' not in data or 'result' not in data['chart'] or not data['chart']['result']:
                print(f"No data available for symbol: {symbol} from {start_date} to {end_date}")
                return pd.DataFrame()

            chart_data = data['chart']['result'][0]
            
            timestamps = chart_data.get('timestamp', [])
            quote_data = chart_data['indicators']['quote'][0]
            
            df = pd.DataFrame({
                'open': quote_data.get('open', []),
                'high': quote_data.get('high', []),
                'low': quote_data.get('low', []),
                'close': quote_data.get('close', []),
                'volume': quote_data.get('volume', []),
            }, index=pd.to_datetime(timestamps, unit='s'))

            # Handle NaN values in the 'volume' column
            df['volume'] = df['volume'].fillna(0).astype('Int64')  # Use 'Int64' to allow for NaN values

            df['symbol'] = symbol

            print(f"Successfully fetched chunk of data for {symbol} from {start_date} to {end_date}. Shape: {df.shape}")
            return df

        except requests.RequestException as e:
            print(f"Error fetching data for {symbol} from {start_date} to {end_date}: {str(e)}")
            print("Full traceback:")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()

    # def get_real_time_quote(self, symbol):
    #     """
    #     Fetch real-time quote for a given symbol.

    #     :param symbol: Stock symbol (e.g., 'AAPL' for Apple Inc.)
    #     :return: Dictionary with real-time quote data
    #     """
    #     print(f"Fetching real-time quote for {symbol}")
    #     url = f"{self.BASE_URL}{symbol}"
    #     params = {
    #         "interval": "1m",
    #         "range": "1d"
    #     }

    #     try:
    #         response = self.session.get(url, params=params)
    #         response.raise_for_status()
    #         data = response.json()

    #         if 'chart' not in data or 'result' not in data['chart'] or not data['chart']['result']:
    #             print(f"No data available for symbol: {symbol}")
    #             return {}

    #         chart_data = data['chart']['result'][0]
    #         meta = chart_data['meta']
            
    #         quote = {
    #             'symbol': symbol,
    #             'price': meta.get('regularMarketPrice'),
    #             'change': meta.get('regularMarketChange'),
    #             'change_percent': meta.get('regularMarketChangePercent'),
    #             'volume': chart_data['indicators']['quote'][0]['volume'][-1] if chart_data['indicators']['quote'][0]['volume'] else None,
    #             'timestamp': datetime.fromtimestamp(meta['regularMarketTime']) if 'regularMarketTime' in meta else None
    #         }
    #         print(f"Successfully fetched real-time quote for {symbol}: {quote}")
    #         return quote

    #     except requests.RequestException as e:
    #         print(f"Error fetching real-time quote for {symbol}: {str(e)}")
    #         print("Full traceback:")
    #         import traceback
    #         traceback.print_exc()
    #         return {}

# Example usage
# if __name__ == "__main__":
#     connector = YahooFinanceConnector()
    
#     # Get historical stock data
#     start_date = datetime.now() - timedelta(days=365*20)  # 20 years ago
#     end_date = datetime.now()
#     print(f"Fetching 20 years of historical data for AAPL")
#     aapl_data = connector.get_stock_data('AAPL', interval='5m', start_date=start_date, end_date=end_date)
#     print(f"AAPL historical data shape: {aapl_data.shape}")
#     print(aapl_data.head())

#     # Get real-time quote
#     print("Fetching real-time quote for AAPL")
#     aapl_quote = connector.get_real_time_quote('AAPL')
#     print(f"AAPL real-time quote: {aapl_quote}")
