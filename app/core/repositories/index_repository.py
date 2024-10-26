from sqlalchemy.future import select
from sqlalchemy import update, delete
from app.core.domain.entities.index import Index, IndexModel
from app.connectors.postgres_client import postgres_client
from typing import List, Optional
from decimal import Decimal
from datetime import datetime
from sqlalchemy.exc import SQLAlchemyError

class IndexRepository:
    async def get_all_indices(self) -> List[Index]:
        with postgres_client.get_session() as session:
            result = session.execute(select(IndexModel))
            return [Index(**row._asdict()) for row in result.scalars().all()]

    async def get_index_by_symbol(self, symbol: str) -> Optional[Index]:
        with postgres_client.get_session() as session:
            result = session.execute(select(IndexModel).filter(IndexModel.symbol == symbol))
            index_model = result.scalar_one_or_none()
            if index_model:
                return Index(**index_model.__dict__)
            return None

    async def add_index(self, index: Index) -> bool:
        with postgres_client.get_session() as session:
            index_model = IndexModel(**index.__dict__)
            session.add(index_model)
            session.commit()
            return True

    async def update_index(self, index: Index) -> bool:
        with postgres_client.get_session() as session:
            result = session.execute(
                update(IndexModel)
                .where(IndexModel.symbol == index.symbol)
                .values(**index.__dict__)
                .returning(IndexModel)
            )
            session.commit()
            return result.rowcount > 0

    async def remove_index(self, symbol: str) -> bool:
        with postgres_client.get_session() as session:
            result = session.execute(
                delete(IndexModel).where(IndexModel.symbol == symbol)
            )
            session.commit()
            return result.rowcount > 0

    async def update_index_price(self, symbol: str, last_price: Decimal, change: Decimal, change_percent: float) -> bool:
        with postgres_client.get_session() as session:
            result = session.execute(
                update(IndexModel)
                .where(IndexModel.symbol == symbol)
                .values(last_price=last_price, change=change, change_percent=change_percent, timestamp=datetime.now())
                .returning(IndexModel)
            )
            session.commit()
            return result.rowcount > 0

    async def get_index_history(self, symbol: str, start_date: datetime, end_date: datetime) -> List[Index]:
        with postgres_client.get_session() as session:
            result = session.execute(
                select(IndexModel)
                .filter(IndexModel.symbol == symbol)
                .filter(IndexModel.timestamp.between(start_date, end_date))
                .order_by(IndexModel.timestamp)
            )
            return [Index(**row._asdict()) for row in result.scalars().all()]

    async def add_price_history_bulk(self, symbol: str, price_data_list: List[Index]) -> bool:
        with postgres_client.get_session() as session:
            try:
                index_models = [
                    IndexModel(
                        symbol=symbol,
                        name=price_data.name,
                        last_price=price_data.last_price,
                        change=price_data.change,
                        change_percent=price_data.change_percent,
                        open=price_data.open,
                        high=price_data.high,
                        low=price_data.low,
                        prev_close=price_data.prev_close,
                        timestamp=price_data.timestamp
                    )
                    for price_data in price_data_list
                ]
                
                session.bulk_save_objects(index_models)
                session.commit()
                print(f"Successfully added {len(index_models)} price history records for index {symbol}")
                return True
            except SQLAlchemyError as e:
                session.rollback()
                print(f"Error adding price history bulk for index {symbol}: {str(e)}")
                return False
