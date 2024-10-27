from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime
from uuid import uuid4
from app.core.domain.models.database_models import (
    TradingSession, Prediction, TradeSuggestion, 
    SessionLog, MarketAnalysis
)

class TradingRepository:
    def __init__(self, db: Session):
        self.db = db
    
    def create_session(self, session_data: dict) -> TradingSession:
        session = TradingSession(
            id=str(uuid4()),
            **session_data
        )
        self.db.add(session)
        self.db.commit()
        self.db.refresh(session)
        return session
    
    def get_session(self, session_id: str) -> Optional[TradingSession]:
        return self.db.query(TradingSession).filter(
            TradingSession.id == session_id
        ).first()
    
    def save_prediction(self, prediction_data: dict) -> Prediction:
        prediction = Prediction(
            id=str(uuid4()),
            **prediction_data
        )
        self.db.add(prediction)
        self.db.commit()
        self.db.refresh(prediction)
        return prediction
    
    def save_suggestion(self, suggestion_data: dict) -> TradeSuggestion:
        suggestion = TradeSuggestion(
            id=str(uuid4()),
            **suggestion_data
        )
        self.db.add(suggestion)
        self.db.commit()
        self.db.refresh(suggestion)
        return suggestion
    
    def log_session_event(self, log_data: dict) -> SessionLog:
        log = SessionLog(**log_data)
        self.db.add(log)
        self.db.commit()
        self.db.refresh(log)
        return log
    
    def save_market_analysis(self, analysis_data: dict) -> MarketAnalysis:
        analysis = MarketAnalysis(
            id=str(uuid4()),
            **analysis_data
        )
        self.db.add(analysis)
        self.db.commit()
        self.db.refresh(analysis)
        return analysis
