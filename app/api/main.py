from fastapi import FastAPI
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

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
