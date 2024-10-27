from sqlalchemy.orm import Session
from sqlalchemy.future import select
from sqlalchemy import update, delete, insert
from app.core.domain.repositories.stock_repository_interface import StockRepositoryInterface
from app.core.domain.entities.stock import Stock, StockPrice, StockModel, StockPriceModel
from app.connectors.postgres_client import postgres_client
from typing import List, Optional, Union, Dict
from decimal import Decimal
from datetime import datetime, timedelta
from sqlalchemy.exc import SQLAlchemyError
from typing import Optional
import logging

class StockRepository(StockRepositoryInterface):
    """Repository for stock data operations"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    async def get_all_stocks(self) -> List[Stock]:
        with postgres_client.get_session() as session:
            result = session.execute(select(StockModel))
            return [Stock(
                symbol=s.symbol,
                name=s.name,
                current_price=s.current_price,
                id=s.id,
                predicted_price=s.predicted_price,
                prediction_version_id=s.prediction_version_id
            ) for s in result.scalars().all()]

    async def get_stock_by_symbol(self, symbol: str) -> Optional[Stock]:
        with postgres_client.get_session() as session:
            result = session.execute(select(StockModel).filter(StockModel.symbol == symbol))
            stock_model = result.scalar_one_or_none()
            if stock_model:
                return Stock(
                    symbol=stock_model.symbol,
                    name=stock_model.name,
                    current_price=stock_model.current_price,
                    id=stock_model.id,
                    predicted_price=stock_model.predicted_price,
                    prediction_version_id=stock_model.prediction_version_id
                )
            return None

    async def update_stock_price(self, symbol: str, price: Decimal) -> bool:
        with postgres_client.get_session() as session:
            result = session.execute(
                update(StockModel)
                .where(StockModel.symbol == symbol)
                .values(current_price=price)
            )
            session.commit()
            return result.rowcount > 0

    async def add_stock(self, stock: Stock) -> bool:
        with postgres_client.get_session() as session:
            stock_model = StockModel(symbol=stock.symbol, name=stock.name, current_price=stock.current_price)
            session.add(stock_model)
            session.commit()
            return True

    async def remove_stock(self, symbol: str) -> bool:
        with postgres_client.get_session() as session:
            result = session.execute(
                delete(StockModel).where(StockModel.symbol == symbol)
            )
            session.commit()
            return result.rowcount > 0

    async def add_price_history_bulk(self, symbol: str, price_data_list: List[StockPrice]) -> bool:
        if not await self.ensure_stock_exists(symbol):
            return False

        with postgres_client.get_session() as session:
            try:
                stock_price_models = [
                    StockPriceModel(
                        stock_symbol=symbol,
                        open=price_data.open,
                        high=price_data.high,
                        low=price_data.low,
                        close=price_data.close,
                        volume=price_data.volume,
                        timestamp=price_data.timestamp
                    )
                    for price_data in price_data_list
                ]
                
                session.bulk_save_objects(stock_price_models)
                session.commit()
                print(f"Successfully added {len(stock_price_models)} price history records for stock {symbol}")
                return True
            except SQLAlchemyError as e:
                session.rollback()
                print(f"Error adding price history bulk for {symbol}: {str(e)}")
                return False

    async def get_price_history(self, symbol: str, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> List[StockPrice]:
        with postgres_client.get_session() as session:
            query = select(StockPriceModel).filter(StockPriceModel.stock_symbol == symbol)
            
            if start_date:
                query = query.filter(StockPriceModel.timestamp >= start_date)
            if end_date:
                query = query.filter(StockPriceModel.timestamp <= end_date)
            
            query = query.order_by(StockPriceModel.timestamp)
            
            result = session.execute(query)
            return [StockPrice(
                open=sp.open,
                high=sp.high,
                low=sp.low,
                close=sp.close,
                volume=sp.volume,
                timestamp=sp.timestamp
            ) for sp in result.scalars().all()]

    async def add_price_history(self, symbol: str, price_data: StockPrice) -> bool:
        with postgres_client.get_session() as session:
            stock_price_model = StockPriceModel(
                stock_symbol=symbol,
                open=price_data.open,
                high=price_data.high,
                low=price_data.low,
                close=price_data.close,
                volume=price_data.volume,
                timestamp=price_data.timestamp
            )
            session.add(stock_price_model)
            session.commit()
            return True

    async def ensure_stock_exists(self, symbol: str, name: str = None) -> bool:
        with postgres_client.get_session() as session:
            stock = session.execute(select(StockModel).filter(StockModel.symbol == symbol)).scalar_one_or_none()
            if not stock:
                new_stock = StockModel(symbol=symbol, name=name or symbol, current_price=0)
                session.add(new_stock)
                try:
                    session.commit()
                    return True
                except SQLAlchemyError as e:
                    session.rollback()
                    print(f"Error ensuring stock exists for {symbol}: {str(e)}")
                    return False
            return True
