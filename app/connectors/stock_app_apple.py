import requests
import pandas as pd
from datetime import datetime, timedelta

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

        :param symbol: Stock symbol (e.g., 'AAPL' for Apple Inc., 'RELIANCE.NS' for Reliance Industries, 'TCS.NS' for Tata Consultancy Services, 'HDFCBANK.NS' for HDFC Bank, 'INFY.NS' for Infosys)
        :param interval: Data interval ('1m', '5m', '15m', '30m', '60m', '1d', '1wk', '1mo')
        :param range: Data range ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', 'max')
        :return: DataFrame with stock data
        """
        url = f"{self.BASE_URL}{symbol}"
        params = {
            "interval": interval,
            "range": range
        }

        response = self.session.get(url, params=params)
        data = response.json()

        if 'chart' not in data or 'result' not in data['chart'] or not data['chart']['result']:
            raise ValueError(f"No data available for symbol: {symbol}")

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

        response = self.session.get(url, params=params)
        data = response.json()

        if 'chart' not in data or 'result' not in data['chart'] or not data['chart']['result']:
            raise ValueError(f"No data available for symbol: {symbol}")

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

    def get_company_info(self, symbol):
        """
        Fetch company information for a given symbol.

        :param symbol: Stock symbol (e.g., 'AAPL' for Apple Inc.)
        :return: Dictionary with company information
        """
        url = f"https://query1.finance.yahoo.com/v10/finance/quoteSummary/{symbol}"
        params = {
            "modules": "assetProfile,summaryDetail,financialData"
        }

        response = self.session.get(url, params=params)
        data = response.json()

        if 'quoteSummary' not in data or 'result' not in data['quoteSummary'] or not data['quoteSummary']['result']:
            raise ValueError(f"No company information available for symbol: {symbol}")

        result = data['quoteSummary']['result'][0]
        
        return {
            'name': result['assetProfile']['longName'],
            'sector': result['assetProfile']['sector'],
            'industry': result['assetProfile']['industry'],
            'website': result['assetProfile']['website'],
            'market_cap': result['summaryDetail']['marketCap']['raw'],
            'pe_ratio': result['summaryDetail']['trailingPE']['raw'],
            'dividend_yield': result['summaryDetail']['dividendYield']['raw'] if 'dividendYield' in result['summaryDetail'] else None,
            '52_week_high': result['summaryDetail']['fiftyTwoWeekHigh']['raw'],
            '52_week_low': result['summaryDetail']['fiftyTwoWeekLow']['raw'],
            'avg_volume': result['summaryDetail']['averageVolume']['raw'],
            'revenue': result['financialData']['totalRevenue']['raw'],
            'gross_profit': result['financialData']['grossProfits']['raw'],
            'ebitda': result['financialData']['ebitda']['raw'],
            'net_income': result['financialData']['netIncomeToCommon']['raw']
        }

# Example usage
if __name__ == "__main__":
    connector = AppleStocksConnector()
    
    # Get historical stock data
    try:
        aapl_data = connector.get_stock_data('RELIANCE.NS', interval='5m', range='1d')
        print(len(aapl_data))
    except ValueError as e:
        print(f"Error fetching historical data: {e}")

    # Get real-time quote
    try:
        aapl_quote = connector.get_real_time_quote('RELIANCE.NS')
        print(aapl_quote)
    except Exception as e:
        print(f"Error fetching real-time quote: {e}")

    # Get company information
    # try:
    #     aapl_info = connector.get_company_info('RELIANCE.NS')
    #     print(aapl_info)
    # except ValueError as e:
    #     print(f"Error fetching company information: {e}")
