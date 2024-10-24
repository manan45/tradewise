import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
from prophet import Prophet
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
    trainer = ppo.PPOTrainer(config=config, env=create_env)
    for i in range(10):
        result = trainer.train()
        print(f"Iteration {i}: reward = {result['episode_reward_mean']}")
    trainer.save("/tmp/ppo_agent")



def generate_trade_suggestions(data: pd.DataFrame) -> list:
    # Ensure the DataFrame has a proper index
    data = data.reset_index(drop=True)

    # Check if the DataFrame is empty
    if data.empty:
        print("Error: Input DataFrame is empty.")
        return []

    # Feature engineering
    data['returns'] = data['close'].pct_change()
    data['ema_9'] = data['close'].ewm(span=9, adjust=False).mean()
    data['ema_12'] = data['close'].ewm(span=12, adjust=False).mean()
    data['ema_15'] = data['close'].ewm(span=15, adjust=False).mean()
    data['ema_26'] = data['close'].ewm(span=26, adjust=False).mean()
    
    # MACD
    macd = MACD(close=data['close'])
    data['macd'] = macd.macd()
    data['signal_line'] = macd.macd_signal()
    
    # RSI
    rsi = RSIIndicator(close=data['close'])
    data['rsi'] = rsi.rsi()
    
    # Bollinger Bands
    bb = BollingerBands(close=data['close'])
    data['upper_band'] = bb.bollinger_hband()
    data['lower_band'] = bb.bollinger_lband()
    
    # Moving Averages
    data['sma_50'] = data['close'].rolling(window=50).mean()
    data['sma_200'] = data['close'].rolling(window=200).mean()
    
    # Stochastic Oscillator
    data['lowest_low'] = data['low'].rolling(window=14).min()
    data['highest_high'] = data['high'].rolling(window=14).max()

