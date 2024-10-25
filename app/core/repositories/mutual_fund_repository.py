from sqlalchemy.future import select
from sqlalchemy import update, delete
from app.core.domain.entities.stock import MutualFund
from app.connectors.postgres_client import postgres_client
from typing import List, Optional

class MutualFundRepository:
    async def get_all_mutual_funds(self) -> List[MutualFund]:
        with postgres_client.get_session() as session:
            result = session.execute(select(MutualFund))
            return result.scalars().all()

    async def get_mutual_fund_by_id(self, fund_id: int) -> Optional[MutualFund]:
        with postgres_client.get_session() as session:
            result = session.execute(select(MutualFund).filter(MutualFund.id == fund_id))
            return result.scalar_one_or_none()

    async def add_mutual_fund(self, mutual_fund: MutualFund) -> bool:
        with postgres_client.get_session() as session:
            session.add(mutual_fund)
            session.flush()
            return True

    async def update_mutual_fund(self, mutual_fund: MutualFund) -> bool:
        with postgres_client.get_session() as session:
            result = session.execute(
                update(MutualFund)
                .where(MutualFund.id == mutual_fund.id)
                .values(name=mutual_fund.name)
                .returning(MutualFund)
            )
            return result.rowcount > 0

    async def remove_mutual_fund(self, fund_id: int) -> bool:
        with postgres_client.get_session() as session:
            result = session.execute(
                delete(MutualFund).where(MutualFund.id == fund_id)
            )
            return result.rowcount > 0
