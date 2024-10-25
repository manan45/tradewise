import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify
import threading
import asyncio
from app.connectors.postgres_client import postgres_client
from app.services.tradewise_ai import TradewiseAI
from app.core.repositories.stock_repository import StockRepository
import pandas as pd
from sqlalchemy import text

load_dotenv()

app = Flask(__name__)

class TrainingService:
    def __init__(self):
        self.stock_repository = StockRepository()
        self.ai = TradewiseAI()

    async def train_model_logic(self):
        try:
            # Fetch stock data from PostgreSQL
            stock_data = await self.stock_repository.get_all_stocks()
            
            # Convert stock data to DataFrame
            df = pd.DataFrame([stock.__dict__ for stock in stock_data])
            
            # Train prediction model
            model = self.ai.train_prediction_model(df)
            
            # Generate predictions
            predictions = self.ai.generate_predictions(model, df)
            
            # Store results back in PostgreSQL
            await self.update_stock_predictions(predictions)
            
            # Generate trade suggestions
            suggestions = self.ai.generate_trade_suggestions(df)
            
            # Store trade suggestions in PostgreSQL
            await self.store_trade_suggestions(suggestions)

            print("Training and prediction completed.")
        except Exception as e:
            print(f"An error occurred during training: {str(e)}")

    async def update_stock_predictions(self, predictions: pd.DataFrame):
        with postgres_client.get_session() as session:
            for _, row in predictions.iterrows():
                sql = text("""
                    UPDATE stocks
                    SET predicted_price = :predicted_price
                    WHERE symbol = :symbol
                """)
                session.execute(sql, {
                    'predicted_price': row['predictions'],
                    'symbol': row['symbol']
                })

    async def store_trade_suggestions(self, suggestions: list):
        with postgres_client.get_session() as session:
            for suggestion in suggestions:
                sql = text("""
                    INSERT INTO trade_suggestions (symbol, action, price, confidence)
                    VALUES (:symbol, :action, :price, :confidence)
                    ON CONFLICT (symbol) DO UPDATE
                    SET action = :action, price = :price, confidence = :confidence
                """)
                session.execute(sql, suggestion)

training_service = TrainingService()

@app.route('/train', methods=['POST'])
def train_model():
    threading.Thread(target=lambda: asyncio.run(training_service.train_model_logic())).start()
    return jsonify({"status": "training started"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv('API_PORT')))
