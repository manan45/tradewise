import pandas as pd
from typing import List
from app.core.domain.entities.stock import Stock, StockPrice
from app.core.repositories.stock_repository import StockRepository
from app.connectors.postgres_client import postgres_client
from datetime import datetime

class FetchAndIngestUseCase:
    def __init__(self, stock_repository: StockRepository):
        self.stock_repository = stock_repository

    def ingest_data_to_postgres(self, data: pd.DataFrame, table_name: str):
        with postgres_client.get_session() as session:
            data.to_sql(table_name, session.bind, if_exists='append', index=True)

    def aggregate_and_ingest(self, dataframe: pd.DataFrame, table_name: str):
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
            self.ingest_data_to_postgres(data, f"{table_name}_{timeframe}")

    async def fetch_and_ingest_stock_data(self, symbol: str, data: pd.DataFrame):
        stock = await self.stock_repository.get_stock_by_symbol(symbol)
        if not stock:
            raise ValueError(f"Stock with symbol {symbol} not found")

        for _, row in data.iterrows():
            price_data = StockPrice(
                open=row['open'],
                high=row['high'],
                low=row['low'],
                close=row['close'],
                volume=row['volume'],
                timestamp=row.name
            )
            await self.stock_repository.add_price_history(symbol, price_data)

        # Update the current price
        latest_price = data.iloc[-1]['close']
        await self.stock_repository.update_stock_price(symbol, latest_price)

        # Aggregate and ingest data
        self.aggregate_and_ingest(data, f"stock_data_{symbol.lower()}")

    async def fetch_and_ingest_multiple_stocks(self, symbols: List[str], data_fetcher):
        for symbol in symbols:
            data = await data_fetcher(symbol)
            await self.fetch_and_ingest_stock_data(symbol, data)

# Example usage:
# async def main():
#     stock_repository = StockRepository()
#     use_case = FetchAndIngestUseCase(stock_repository)
#     
#     async def mock_data_fetcher(symbol):
#         return pd.DataFrame({
#             'timestamp': pd.date_range(start='2023-01-01', end='2023-01-02', freq='1min'),
#             'open': [100] * 1440,
#             'high': [110] * 1440,
#             'low': [90] * 1440,
#             'close': [105] * 1440,
#             'volume': [1000] * 1440
#         })
#     
#     await use_case.fetch_and_ingest_multiple_stocks(['AAPL', 'GOOGL', 'MSFT'], mock_data_fetcher)
# 
# if __name__ == "__main__":
#     import asyncio
#     asyncio.run(main())
