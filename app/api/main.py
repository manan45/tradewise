from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import asyncio
from fastapi.openapi.utils import get_openapi
from utils.data_loader import load_stock_data, get_latest_stock_data
from utils.ai_model import generate_trade_suggestions
from pydantic import BaseModel

app = FastAPI(title="Stock Trading API",
              description="API for stock trading suggestions and real-time data",
              version="1.0.0")

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

@app.post("/trade-suggestions")
def get_trade_suggestions(request: TradeSuggestionRequest):
    data = load_stock_data(request.symbol, request.date)
    suggestions = generate_trade_suggestions(data)
    return suggestions

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            await asyncio.sleep(5)  # Simulate real-time data fetching every 5 seconds
            latest_data = get_latest_stock_data()
            suggestions = generate_trade_suggestions(latest_data)
            await websocket.send_json(suggestions[0].dict())
    except WebSocketDisconnect:
        print("WebSocket disconnected")

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(fetch_real_time_data())

async def fetch_real_time_data():
    while True:
        try:
            # Simulate fetching real-time data
            await asyncio.sleep(5)  # Fetch data every 5 seconds
            print("Fetching real-time data...")
        except Exception as e:
            print(f"Error fetching real-time data: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
