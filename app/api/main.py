import os
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from app.services.data_service import DataService
from app.services.tradewise_ai import TradewiseAI
from typing import List
from app.core.domain.models.trade_suggestion import DetailedTradeSuggestion
from app.core.use_cases.trade_suggestions import TradeSuggestionsService, TradeSuggestionsUseCase
from app.core.repositories.stock_repository import StockRepository
from app.connectors.postgres_client import postgres_client, get_db
from app.core.domain.models.trade_suggestion_request import TradeSuggestionRequest
from sqlalchemy.orm import Session
from app.connectors.yahoo_finance import AppleStocksConnector
from app.core.use_cases.fetch_option_data import FetchOptionDataUseCase
from datetime import datetime

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




@app.post("/trade-suggestions", response_model=List[DetailedTradeSuggestion])
async def get_trade_suggestions(request: TradeSuggestionRequest, db: Session = Depends(get_db)):
    stock_repository = StockRepository(db)
    trade_suggestions_service = TradeSuggestionsService(stock_repository)
    trade_suggestions = TradeSuggestionsUseCase(stock_repository, trade_suggestions_service)
    suggestions = await trade_suggestions.generate_suggestion_for_stock(request.symbol)
    return suggestions

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

@app.get("/api/option-data/{symbol}")
async def get_option_data(symbol: str, expiry_date: str):
    stock_repository = StockRepository()
    fetch_option_data_use_case = FetchOptionDataUseCase(stock_repository)
    expiry = datetime.strptime(expiry_date, "%Y-%m-%d")
    
    option_data = await fetch_option_data_use_case.execute(symbol, expiry)
    
    return option_data

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=os.getenv('API_HOST'), port=int(os.getenv('API_PORT')))
