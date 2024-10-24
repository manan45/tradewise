from fastapi import FastAPI, WebSocket
import asyncio
from app.core.utils.hello import hello
from app.core.use_cases.trade_suggestions import TradeSuggestions
from app.core.interface_adapters.dhan import DhanAPI

app = FastAPI()

@app.get("/trade-suggestions")
def get_trade_suggestions():
    dhan_api = DhanAPI(api_key="your_api_key")
    trade_suggestions = TradeSuggestions(stock_repository=dhan_api)
    suggestions = trade_suggestions.generate_suggestions()
    return suggestions

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        # Process incoming data and send response
        await websocket.send_text(f"Received: {data}")
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(fetch_real_time_data())

async def fetch_real_time_data():
    while True:
        # Simulate fetching real-time data
        await asyncio.sleep(5)  # Fetch data every 5 seconds
        print("Fetching real-time data...")
