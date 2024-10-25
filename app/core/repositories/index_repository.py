from sqlalchemy.future import select
from sqlalchemy import update, delete
from app.core.domain.entities.stock import Index
from app.connectors.postgres_client import postgres_client
from typing import List, Optional

class IndexRepository:
    async def get_all_indices(self) -> List[Index]:
        with postgres_client.get_session() as session:
            result = session.execute(select(Index))
            return result.scalars().all()

    async def get_index_by_id(self, index_id: int) -> Optional[Index]:
        with postgres_client.get_session() as session:
            result = session.execute(select(Index).filter(Index.id == index_id))
            return result.scalar_one_or_none()

    async def add_index(self, index: Index) -> bool:
        with postgres_client.get_session() as session:
            session.add(index)
            session.flush()
            return True

    async def update_index(self, index: Index) -> bool:
        with postgres_client.get_session() as session:
            result = session.execute(
                update(Index)
                .where(Index.id == index.id)
                .values(name=index.name)
                .returning(Index)
            )
            return result.rowcount > 0

    async def remove_index(self, index_id: int) -> bool:
        with postgres_client.get_session() as session:
            result = session.execute(
                delete(Index).where(Index.id == index_id)
            )
            return result.rowcount > 0
