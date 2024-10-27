import os
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from typing import List, Optional
from datetime import datetime

from app.core.domain.models.trade_suggestion_request import TradeSuggestionRequest
from app.core.use_cases.trade_suggestions import TradeSuggestionsUseCase
from app.core.repositories.stock_repository import StockRepository
from app.core.ai.tradewise_ai import TradewiseAI, SessionStats, PredictionStats
from app.services.data_service import DataService

load_dotenv()

app = FastAPI(
    title="Stock Trading API",
    description="API for stock trading suggestions and real-time data",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="Stock Trading API",
        version="1.0.0",
        description="API for stock trading suggestions and real-time data",
        routes=app.routes,
    )
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

@app.post("/api/tradewise/suggestions")
async def generate_trade_suggestions(request: TradeSuggestionRequest):
    """Generate trade suggestions for a symbol"""
    try:
        stock_repository = StockRepository()
        use_case = TradeSuggestionsUseCase(stock_repository)
        suggestion = await use_case.generate_suggestion_for_stock(request.symbol)
        return suggestion
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/tradewise/suggestions/top/{limit}")
async def get_top_suggestions(limit: int = 5):
    """Get top trade suggestions"""
    try:
        stock_repository = StockRepository()
        use_case = TradeSuggestionsUseCase(stock_repository)
        suggestions = await use_case.get_top_suggestions(limit)
        return suggestions
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/tradewise/sessions/{session_id}")
async def get_session_stats(session_id: Optional[str] = None):
    """Get trading session statistics"""
    try:
        tradewise = TradewiseAI()
        stats = await tradewise.get_session_stats(session_id)
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/tradewise/predictions/{session_id}")
async def get_prediction_stats(session_id: Optional[str] = None):
    """Get prediction statistics"""
    try:
        tradewise = TradewiseAI()
        stats = await tradewise.get_prediction_stats(session_id)
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/tradewise/logs")
async def get_session_logs(
    session_id: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
):
    """Get trading session logs"""
    try:
        tradewise = TradewiseAI()
        logs = await tradewise.get_session_logs(session_id, start_date, end_date)
        return logs
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/tradewise/analysis/{symbol}")
async def get_market_analysis(symbol: str):
    """Get comprehensive market analysis for a symbol"""
    try:
        stock_repository = StockRepository()
        tradewise = TradewiseAI()
        df = await stock_repository.get_market_data(symbol)
        analysis = tradewise._analyze_market(df)
        return analysis
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data_service = DataService()
            tradewise_ai = TradewiseAI()
            latest_data = await data_service.get_latest_stock_data()
            suggestions = await tradewise_ai.generate_trade_suggestions(latest_data)
            await websocket.send_json(suggestions[0].dict())
    except WebSocketDisconnect:
        print("WebSocket disconnected")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=os.getenv('API_HOST'), port=int(os.getenv('API_PORT')))
