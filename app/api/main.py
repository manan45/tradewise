from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from app.utils.data_loader import load_stock_data, get_latest_stock_data
from app.utils.ai_model import generate_trade_suggestions
from pydantic import BaseModel
from typing import List
from app.core.domain.models import DetailedTradeSuggestion
from sqlalchemy.orm import Session
from app.core.database import get_db_session

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

class TradeSuggestionRequest(BaseModel):
    symbol: str
    date: str

@app.post("/trade-suggestions", response_model=List[DetailedTradeSuggestion])
def get_trade_suggestions(request: TradeSuggestionRequest):
    with get_db_session() as db:
        data = load_stock_data(request.symbol, request.date, db)
        suggestions = generate_trade_suggestions(data)
        return suggestions

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            latest_data = get_latest_stock_data()
            suggestions = generate_trade_suggestions(latest_data)
            await websocket.send_json(suggestions[0].dict())
    except WebSocketDisconnect:
        print("WebSocket disconnected")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
