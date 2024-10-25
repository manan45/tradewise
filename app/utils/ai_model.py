import pandas as pd
import numpy as np
from utils.forecasting import lstm_forecast, forecast_timeseries
from core.domain.models import DetailedTradeSuggestion
from sklearn.ensemble import RandomForestRegressor
from ta.trend import MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
import gym
from gym import spaces

class StockTradingEnv(gym.Env):
    def __init__(self, data):
        super(StockTradingEnv, self).__init__()
        self.data = data
        self.reward_range = (-np.inf, np.inf)
        self.action_space = spaces.Discrete(3)  # 0: Hold, 1: Buy, 2: Sell
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)
        self.reset()

    def reset(self):
        self.current_step = 0
        self.total_reward = 0
        self.holdings = 0
        self.cash = 10000  # Starting cash
        return self._next_observation()

    def _next_observation(self):
        obs = np.array([
            self.data['close'].iloc[self.current_step],
            self.data['macd'].iloc[self.current_step],
            self.data['rsi'].iloc[self.current_step],
            self.data['bb_high'].iloc[self.current_step],
            self.data['bb_low'].iloc[self.current_step]
        ])
        return obs

    def step(self, action):
        self.current_step += 1
        current_price = self.data['close'].iloc[self.current_step]
        reward = 0

        if action == 1:  # Buy
            shares_to_buy = self.cash // current_price
            self.holdings += shares_to_buy
            self.cash -= shares_to_buy * current_price
        elif action == 2:  # Sell
            reward = self.holdings * current_price - self.holdings * self.data['close'].iloc[self.current_step - 1]
            self.cash += self.holdings * current_price
            self.holdings = 0

        done = self.current_step >= len(self.data) - 1
        obs = self._next_observation()
        return obs, reward, done, {}

def create_env(data):
    return StockTradingEnv(data)

def train_agent(env, n_estimators=100, random_state=42):
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    training_data = []
    training_labels = []

    for _ in range(1000):  # Run 1000 episodes
        obs = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()  # Random action for data collection
            next_obs, reward, done, _ = env.step(action)
            training_data.append(obs)
            training_labels.append(reward)
            obs = next_obs

    model.fit(training_data, training_labels)
    return model

def generate_trade_suggestions(data: pd.DataFrame) -> list:
    # Prepare features
    macd = MACD(close=data['close'])
    rsi = RSIIndicator(close=data['close'])
    bb = BollingerBands(close=data['close'])

    data['macd'] = macd.macd()
    data['rsi'] = rsi.rsi()
    data['bb_high'] = bb.bollinger_hband()
    data['bb_low'] = bb.bollinger_lband()

    env = create_env(data)
    model = train_agent(env)

    suggestions = []
    obs = env.reset()
    for _ in range(5):  # Generate suggestions for the next 5 steps
        action = model.predict([obs])[0]
        if action == 1:
            suggestions.append(DetailedTradeSuggestion(action="BUY", price=data['close'].iloc[env.current_step], confidence=0.8))
        elif action == 2:
            suggestions.append(DetailedTradeSuggestion(action="SELL", price=data['close'].iloc[env.current_step], confidence=0.8))
        obs, _, done, _ = env.step(action)
        if done:
            break

    return suggestions
