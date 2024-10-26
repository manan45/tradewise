from decimal import Decimal
import os
from textwrap import indent
import time
from dotenv import load_dotenv
from app.connectors.yahoo_finance import YahooFinanceConnector
from app.connectors.dhanhq.dhanhq import dhanhq
from app.core.repositories.index_repository import IndexRepository
from app.core.use_cases.fetch_and_ingest import FetchAndIngestUseCase
from app.core.repositories.stock_repository import StockRepository
from app.services.base_data_service import BaseDataService
import pandas as pd
from datetime import datetime, timedelta
import asyncio
from app.core.domain.entities.stock import Index, StockPrice
from app.core.domain.entities.option_data import OptionChain, OpenInterest
from typing import List, Dict, Any
from app.core.domain.enums.dhan_enums import ExchangeSegment, ProductType, Instrument, ExpiryCode

class YahooDataService(BaseDataService):
    def __init__(self, index: str, stocks: List[str]):
        super().__init__()
        self.stocks = stocks
        self.index = index
        self.connector = YahooFinanceConnector()
        print(f"Initialized YahooDataService for index: {index} and stocks: {stocks}")

    def fetch_historical_data(self, interval: str, start_date: datetime, end_date: datetime) -> Dict[str, pd.DataFrame]:
        print(f"Fetching historical data for {self.index} and stocks with interval: {interval}, from {start_date} to {end_date}")
        try:
            data = {}
            if self.index:
                index_df = self.connector.get_stock_data(self.index, interval, start_date, end_date)
                if not index_df.empty:
                    data['index'] = index_df
                    print(f"Successfully fetched historical data for index {self.index}. Shape: {index_df.shape}")
                else:
                    print(f"No historical data available for index {self.index}")

            if self.stocks:
                stocks_df = pd.DataFrame()
                for stock in self.stocks:
                    stock_data = self.connector.get_stock_data(stock, interval, start_date, end_date)
                    if not stock_data.empty:
                        stocks_df = pd.concat([stocks_df, stock_data])
                        print(f"Successfully fetched historical data for stock {stock}. Shape: {stock_data.shape}")
                    else:
                        print(f"No historical data available for stock {stock}")
                if not stocks_df.empty:
                    data['stocks'] = stocks_df
                else:
                    print(f"No historical data available for any of the stocks")

            return data
        except Exception as e:
            print(f"Failed to fetch historical data: {str(e)}")
            raise

    async def fetch_and_process_data(self):
        print(f"Starting fetch_and_process_data for index: {self.index} and stocks: {self.stocks}")

        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=7)  # Fetch data for the last 20 years
            data = self.fetch_historical_data(interval='1m', start_date=start_date, end_date=end_date)
            
            # Process index data
            if 'index' in data:
                index_prices = [
                    Index(
                        symbol=self.index,
                        name=self.index,
                        last_price=Decimal(str(row['close'])),
                        change=Decimal(str(row['close'] - row['open'])),
                        change_percent=float((row['close'] - row['open']) / row['open'] * 100),
                        open=Decimal(str(row['open'])),
                        high=Decimal(str(row['high'])),
                        low=Decimal(str(row['low'])),
                        prev_close=Decimal(str(row['close'])),
                        timestamp=row.name
                    )
                    for _, row in data['index'].iterrows()
                ]
                await self.index_repository.add_price_history_bulk(self.index, index_prices)
                print(f"Finished adding price history bulk for index {self.index}")
            else:
                print(f"No index data available for {self.index}")

            # Process stock data
            if 'stocks' in data:
                for symbol in self.stocks:
                    stock_data = data['stocks'][data['stocks']['symbol'] == symbol]
                    if not stock_data.empty:
                        # Ensure the stock exists before adding price data
                        await self.stock_repository.ensure_stock_exists(symbol)
                        
                        stock_prices = [
                            StockPrice(
                                open=Decimal(str(row['open'])),
                                high=Decimal(str(row['high'])),
                                low=Decimal(str(row['low'])),
                                close=Decimal(str(row['close'])),
                                volume=int(row['volume']),
                                timestamp=row.name
                            )
                            for _, row in stock_data.iterrows()
                        ]
                        await self.stock_repository.add_price_history_bulk(symbol, stock_prices)
                        print(f"Finished adding price history bulk for stock {symbol}")
                    else:
                        print(f"No data available for stock {symbol}")
            else:
                print("No stock data available")

        except Exception as e:
            print(f"Error fetching or processing data: {str(e)}")
            raise

        print(f"Completed fetch_and_process_data for index: {self.index} and stocks: {self.stocks}")

# class DhanDataService(BaseDataService):
#     def __init__(self, symbol: str):
#         super().__init__()
#         self.client_id = os.getenv('DHAN_CLIENT_ID')
#         self.access_token = os.getenv('DHAN_ACCESS_TOKEN')
#         self.symbol = symbol
#         if not self.client_id or not self.access_token:
#             raise ValueError("DHAN_CLIENT_ID and DHAN_ACCESS_TOKEN must be set in the environment or .env file")
#         self.dhan_api = dhanhq(client_id=self.client_id, access_token=self.access_token)
#         self.stock_repository = StockRepository()
#         print(f"Initialized DhanDataService for symbol: {symbol}")

#     def fetch_historical_data(self, start_date: datetime, end_date: datetime, interval='1d') -> pd.DataFrame:
#         try:
#             print(f"Fetching historical data for {self.symbol} from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
#             historical_data = self.dhan_api.intraday_minute_data(
#                 security_id='NIFTY50',
#                 exchange_segment='NSE_EQ',
#                 instrument_type='EQUITY',
#                 from_date=start_date.strftime('%Y-%m-%d'),
#                 to_date=end_date.strftime('%Y-%m-%d'),
#                 interval=1
#             )
            
#             if historical_data['status'] == 'success':
#                 df = pd.DataFrame(historical_data['data'])
#                 df['timestamp'] = pd.to_datetime(df['date'])
#                 df = df.drop('date', axis=1)
#                 print(f"Successfully fetched historical data for {self.symbol}. Shape: {df.shape}")
#                 return df
#             else:
#                 error_message = historical_data.get('remarks', 'Unknown error')
#                 error_code = historical_data.get('error_code', 'Unknown error code')
#                 print(f"Failed to fetch historical data for {self.symbol}. Error code: {error_code}, Message: {error_message}")
#                 return pd.DataFrame()
#         except Exception as e:
#             print(f"Exception while fetching historical data for {self.symbol}: {str(e)}")
#             return pd.DataFrame()

#     def fetch_nifty_components(self) -> List[Dict]:
#         try:
#             nifty_components = self.dhan_api.quote_data({ExchangeSegment.NSE_EQ.name: ['NIFTY50']})
#             if nifty_components['status'] == 'success':
#                 return nifty_components['data'][ExchangeSegment.NSE_EQ.name]
#             else:
#                 print(f"Failed to fetch Nifty components: {nifty_components['remarks']}")
#                 return []
#         except Exception as e:
#             print(f"Exception while fetching Nifty components: {str(e)}")
#             return []

#     def fetch_option_chain(self, expiry_date: datetime) -> List[Dict]:
#         try:
#             option_chain = self.dhan_api.option_chain(
#                 under_security_id=self.symbol,
#                 under_exchange_segment=ExchangeSegment.NSE_FNO.name,
#                 expiry=expiry_date.strftime('%Y-%m-%d')
#             )
#             if option_chain['status'] == 'success':
#                 return option_chain['data']
#             else:
#                 print(f"Failed to fetch option chain for {self.symbol}: {option_chain['remarks']}")
#                 return []
#         except Exception as e:
#             print(f"Exception while fetching option chain for {self.symbol}: {str(e)}")
#             return []

#     def fetch_open_interest(self, expiry_date: datetime, strike_price: float, option_type: str) -> Dict:
#         try:
#             oi_data = self.dhan_api.quote_data({
#                 ExchangeSegment.NSE_FNO.name: [f"{self.symbol}{expiry_date.strftime('%d%b%y').upper()}{strike_price}{option_type}"]
#             })
#             if oi_data['status'] == 'success':
#                 return oi_data['data'][ExchangeSegment.NSE_FNO.name][0]
#             else:
#                 print(f"Failed to fetch OI data for {self.symbol} {strike_price} {option_type}: {oi_data['remarks']}")
#                 return {}
#         except Exception as e:
#             print(f"Exception while fetching OI data for {self.symbol} {strike_price} {option_type}: {str(e)}")
#             return {}

#     async def fetch_and_process_data(self):
#         print(f"Starting fetch_and_process_data for symbol: {self.symbol}")
        
#         # Fetch Nifty index data
#         end_date = datetime.now()
#         start_date = end_date - timedelta(days=30)  # Last 30 days of data
#         nifty_data = self.fetch_historical_data(start_date, end_date)
        
#         # Fetch Nifty components
#         nifty_components = self.fetch_nifty_components()
        
#         # Fetch option chain data
#         next_expiry = self.dhan_api.expiry_list(self.symbol, ExchangeSegment.NSE_FNO.name)['data'][ExpiryCode.NEXT.value]
#         option_chain = self.fetch_option_chain(datetime.strptime(next_expiry, '%Y-%m-%d'))
        
#         # Fetch OI data for ATM, ITM, and OTM options
#         current_price = nifty_data['close'].iloc[-1]
#         atm_strike = round(current_price / 50) * 50  # Round to nearest 50
#         strikes_to_fetch = [atm_strike - 100, atm_strike, atm_strike + 100]
        
#         oi_data = []
#         for strike in strikes_to_fetch:
#             for option_type in ['CE', 'PE']:
#                 oi = self.fetch_open_interest(datetime.strptime(next_expiry, '%Y-%m-%d'), strike, option_type)
#                 if oi:
#                     oi_data.append(oi)
        
#         # Store the data in the repository
#         await self.stock_repository.add_price_history_bulk(self.symbol, nifty_data.to_dict('records'))
#         await self.stock_repository.add_option_chain_data(option_chain)
#         for oi in oi_data:
#             await self.stock_repository.add_open_interest_data(oi)
        
#         print(f"Completed fetch_and_process_data for symbol: {self.symbol}")
        
#         return {
#             "nifty_data": nifty_data.to_dict('records'),
#             "nifty_components": nifty_components,
#             "option_chain": option_chain,
#             "oi_data": oi_data
#         }

def get_data_service(data_source: str, symbol: str = None, security_id: str = None):
    if data_source == 'apple':
        # index = 'DJI'  # Example symbols for Apple
        stocks = ['AAPL']
        return YahooDataService(index = None, stocks = stocks)      
    elif data_source == 'dhan':
        # return DhanDataService(symbol)
        pass
    else:
        raise ValueError(f"Invalid DATA_SOURCE: {data_source}. Must be 'apple' or 'dhan'.")

async def main():
    load_dotenv()
    data_source = os.getenv('DATA_SOURCE', 'apple').lower()
    print(f"Starting data service with data source: {data_source}")
    connector = get_data_service(data_source)
    try:
        print("Starting fetch and process data")
        await connector.fetch_and_process_data()
        print("Completed fetch and process data")
    except Exception as e:
        print(f"An error occurred in main: {str(e)}")

if __name__ == "__main__":
    print("Starting data service script")
    asyncio.run(main())
    print("Data service script completed")
