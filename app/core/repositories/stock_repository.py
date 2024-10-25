from sqlalchemy.orm import Session
from sqlalchemy.future import select
from sqlalchemy import update, delete, insert
from app.core.domain.repositories.stock_repository_interface import StockRepositoryInterface
from app.core.domain.entities.stock import Stock, StockPrice, StockModel, StockPriceModel
from app.connectors.postgres_client import postgres_client
from typing import List, Optional
from decimal import Decimal
from datetime import datetime, timedelta

class StockRepository(StockRepositoryInterface):
    async def get_all_stocks(self) -> List[Stock]:
        with postgres_client.get_session() as session:
            result = session.execute(select(StockModel))
            return [Stock(symbol=s.symbol, name=s.name, current_price=s.current_price, id=s.id) for s in result.scalars().all()]

    async def get_stock_by_symbol(self, symbol: str) -> Optional[Stock]:
        with postgres_client.get_session() as session:
            result = session.execute(select(StockModel).filter(StockModel.symbol == symbol))
            stock_model = result.scalar_one_or_none()
            if stock_model:
                return Stock(symbol=stock_model.symbol, name=stock_model.name, current_price=stock_model.current_price, id=stock_model.id)
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

    async def get_price_history(self, symbol: str, days: int = 30) -> List[StockPrice]:
        with postgres_client.get_session() as session:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            result = session.execute(
                select(StockPriceModel)
                .filter(StockPriceModel.stock_symbol == symbol)
                .filter(StockPriceModel.timestamp.between(start_date, end_date))
                .order_by(StockPriceModel.timestamp)
            )
            return [StockPrice(
                open=sp.open,
                high=sp.high,
                low=sp.low,
                close=sp.close,
                volume=sp.volume,
                timestamp=sp.timestamp
            ) for sp in result.scalars().all()]
