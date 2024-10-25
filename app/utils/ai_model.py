import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
from prophet import Prophet
from app.utils.forecasting import lstm_forecast, forecast_timeseries
from app.core.domain.models import DetailedTradeSuggestion
from sklearn.ensemble import RandomForestRegressor
from ta.trend import MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
import gym
import ray
from ray.rllib.agents import ppo

ray.init(ignore_reinit_error=True)

def create_env():
    # Define a custom environment or use an existing one
    return gym.make('CartPole-v1')

def train_agent():
    config = ppo.DEFAULT_CONFIG.copy()
    config["num_workers"] = 1
    config["framework"] = "torch"  # Use PyTorch for better performance
    config["train_batch_size"] = 4000
    config["sgd_minibatch_size"] = 128
    config["num_sgd_iter"] = 30
    trainer = ppo.PPOTrainer(config=config, env=create_env)
    for i in range(10):
        result = trainer.train()
        print(f"Iteration {i}: reward = {result['episode_reward_mean']}")
    trainer.save("/tmp/ppo_agent")



def generate_trade_suggestions(data: pd.DataFrame) -> list:
    # Prepare features
    data['returns'] = data['close'].pct_change()
    data['sma_5'] = data['close'].rolling(window=5).mean()
    data['sma_20'] = data['close'].rolling(window=20).mean()
    
    # Drop NaN values
    data = data.dropna()
    
    # Prepare features and target
    features = ['open', 'high', 'low', 'close', 'volume', 'returns', 'sma_5', 'sma_20', 'news_sentiment']
    X = data[features]
    y = data['close'].shift(-1)  # Predict next day's close price
    
    # Remove the last row as we don't have the next day's price for it
    X = X[:-1]
    y = y[:-1]
    
    # Train the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Make predictions for the next day
    last_data = X.iloc[-1].values.reshape(1, -1)
    prediction = model.predict(last_data)[0]
    
    # Generate trade suggestion
    current_price = data['close'].iloc[-1]
    action = "BUY" if prediction > current_price else "SELL"
    confidence = abs(prediction - current_price) / current_price
    
    suggestion = DetailedTradeSuggestion(
        action=action,
        price=prediction,
        confidence=confidence,
        stop_loss=min(data['low'].iloc[-5:]),
        order_limit=max(data['high'].iloc[-5:]),
        max_risk=0.02 * current_price,
        max_reward=0.05 * current_price,
        open=data['open'].iloc[-1],
        high=data['high'].iloc[-1],
        low=data['low'].iloc[-1],
        close=current_price
    )
    
    return [suggestion]

def train_agent():
    # Placeholder for future implementation
    pass
