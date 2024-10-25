from sqlalchemy.orm import Session
from sqlalchemy.future import select
from sqlalchemy import update, delete
from app.core.domain.repositories.stock_repository_interface import StockRepositoryInterface
from app.core.domain.entities.stock import Stock, StockPrice
from app.connectors.postgres_client import postgres_client
from typing import List, Optional
from decimal import Decimal
from datetime import datetime, timedelta

class StockRepository(StockRepositoryInterface):
    async def get_all_stocks(self) -> List[Stock]:
        with postgres_client.get_session() as session:
            result = session.execute(select(Stock))
            return result.scalars().all()

    async def get_stock_by_symbol(self, symbol: str) -> Optional[Stock]:
        with postgres_client.get_session() as session:
            result = session.execute(select(Stock).filter(Stock.symbol == symbol))
            return result.scalar_one_or_none()

    async def update_stock_price(self, symbol: str, price: Decimal) -> bool:
        with postgres_client.get_session() as session:
            result = session.execute(
                update(Stock)
                .where(Stock.symbol == symbol)
                .values(current_price=price)
                .returning(Stock)
            )
            return result.rowcount > 0

    async def add_stock(self, stock: Stock) -> bool:
        with postgres_client.get_session() as session:
            session.add(stock)
            session.flush()
            return True

    async def remove_stock(self, symbol: str) -> bool:
        with postgres_client.get_session() as session:
            result = session.execute(
                delete(Stock).where(Stock.symbol == symbol)
            )
            return result.rowcount > 0

    async def add_price_history(self, symbol: str, price_data: StockPrice) -> bool:
        with postgres_client.get_session() as session:
            stock = await self.get_stock_by_symbol(symbol)
            if stock:
                stock.add_price_history(price_data)
                session.flush()
                return True
            return False

    async def get_price_history(self, symbol: str, days: int = 30) -> List[StockPrice]:
        with postgres_client.get_session() as session:
            stock = await self.get_stock_by_symbol(symbol)
            if stock:
                end_date = datetime.now()
                start_date = end_date - timedelta(days=days)
                return [price for price in stock.price_history if start_date <= price.timestamp <= end_date]
            return []
