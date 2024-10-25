import requests
import pandas as pd
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class AppleStocksConnector:
    BASE_URL = "https://query1.finance.yahoo.com/v8/finance/chart/"

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        })

    def get_stock_data(self, symbol, interval='1d', range='1mo'):
        """
        Fetch stock data for a given symbol.

        :param symbol: Stock symbol (e.g., 'AAPL' for Apple Inc.)
        :param interval: Data interval ('1m', '5m', '15m', '30m', '60m', '1d', '1wk', '1mo')
        :param range: Data range ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', 'max')
        :return: DataFrame with stock data
        """
        url = f"{self.BASE_URL}{symbol}"
        params = {
            "interval": interval,
            "range": range
        }

        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()  # Raise an exception for bad status codes
            data = response.json()

            if 'chart' not in data or 'result' not in data['chart'] or not data['chart']['result']:
                logger.error(f"No data available for symbol: {symbol}")
                return pd.DataFrame()  # Return an empty DataFrame instead of raising an exception

            chart_data = data['chart']['result'][0]
            
            df = pd.DataFrame({
                'timestamp': pd.to_datetime(chart_data['timestamp'], unit='s'),
                'open': chart_data['indicators']['quote'][0]['open'],
                'high': chart_data['indicators']['quote'][0]['high'],
                'low': chart_data['indicators']['quote'][0]['low'],
                'close': chart_data['indicators']['quote'][0]['close'],
                'volume': chart_data['indicators']['quote'][0]['volume']
            })

            df.set_index('timestamp', inplace=True)
            return df

        except requests.RequestException as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return pd.DataFrame()  # Return an empty DataFrame on error

    def get_real_time_quote(self, symbol):
        """
        Fetch real-time quote for a given symbol.

        :param symbol: Stock symbol (e.g., 'AAPL' for Apple Inc.)
        :return: Dictionary with real-time quote data
        """
        url = f"{self.BASE_URL}{symbol}"
        params = {
            "interval": "1m",
            "range": "1d"
        }

        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()  # Raise an exception for bad status codes
            data = response.json()

            if 'chart' not in data or 'result' not in data['chart'] or not data['chart']['result']:
                logger.error(f"No data available for symbol: {symbol}")
                return {}  # Return an empty dictionary instead of raising an exception

            chart_data = data['chart']['result'][0]
            meta = chart_data['meta']
            
            return {
                'symbol': symbol,
                'price': meta.get('regularMarketPrice'),
                'change': meta.get('regularMarketChange'),
                'change_percent': meta.get('regularMarketChangePercent'),
                'volume': chart_data['indicators']['quote'][0]['volume'][-1] if chart_data['indicators']['quote'][0]['volume'] else None,
                'timestamp': datetime.fromtimestamp(meta['regularMarketTime']) if 'regularMarketTime' in meta else None
            }

        except requests.RequestException as e:
            logger.error(f"Error fetching real-time quote for {symbol}: {str(e)}")
            return {}  # Return an empty dictionary on error

# Example usage
if __name__ == "__main__":
    connector = AppleStocksConnector()
    
    # Get historical stock data
    aapl_data = connector.get_stock_data('AAPL', interval='1d', range='1mo')
    print(aapl_data)

    # Get real-time quote
    aapl_quote = connector.get_real_time_quote('AAPL')
    print(aapl_quote)
