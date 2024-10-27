import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any

class MarketEnvironment(gym.Env):
    """Custom Market Environment for RL training"""
    
    def __init__(self, data: pd.DataFrame, window_size: int = 60):
        super(MarketEnvironment, self).__init__()
        
        self.data = data
        self.window_size = window_size
        self.current_step = 0
        
        # Define action space (0: Sell, 1: Hold, 2: Buy)
        self.action_space = spaces.Discrete(3)
        
        # Define observation space
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(window_size, len(self.data.columns)),
            dtype=np.float32
        )
        
        # Trading params
        self.initial_balance = 10000
        self.balance = self.initial_balance
        self.position = 0
        self.trades = []

    def reset(self, seed=None) -> Tuple[np.ndarray, Dict]:
        """Reset environment"""
        super().reset(seed=seed)
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.position = 0
        self.trades = []
        
        return self._get_observation(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Take action in environment"""
        # Get current price
        current_price = self.data.iloc[self.current_step]['close']
        
        # Execute action
        reward = self._execute_action(action, current_price)
        
        # Move to next step
        self.current_step += 1
        
        # Check if done
        done = self.current_step >= len(self.data) - 1
        
        # Get next observation
        next_obs = self._get_observation()
        
        return next_obs, reward, done, False, {'balance': self.balance, 'position': self.position}

    def _get_observation(self) -> np.ndarray:
        """Get current market observation"""
        return self.data.iloc[self.current_step-self.window_size:self.current_step].values

    def _execute_action(self, action: int, price: float) -> float:
        """Execute trading action and calculate reward"""
        prev_balance = self.balance
        
        if action == 0:  # Sell
            if self.position > 0:
                self.balance += self.position * price
                self.trades.append({
                    'type': 'sell',
                    'price': price,
                    'quantity': self.position,
                    'balance': self.balance
                })
                self.position = 0
                
        elif action == 2:  # Buy
            if self.position == 0:
                quantity = self.balance / price
                self.position = quantity
                self.balance = 0
                self.trades.append({
                    'type': 'buy',
                    'price': price,
                    'quantity': quantity,
                    'balance': self.balance
                })
        
        # Calculate reward (change in portfolio value)
        portfolio_value = self.balance + (self.position * price)
        reward = (portfolio_value - prev_balance) / prev_balance
        
        return reward

    def render(self):
        """Render environment"""
        pass
