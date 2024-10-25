from sqlalchemy.orm import Session
from app.core.entities.entities import Stock

class StockRepository:
    def __init__(self, db: Session):
        self.db = db

    async def get_all_stocks(self):
        return self.db.query(Stock).all()
