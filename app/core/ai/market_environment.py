import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any, List

from app.core.ai.reinforcement.action_space import ActionSpace
from app.core.ai.reinforcement.market_state import MarketState
from app.core.ai.reinforcement.reward_calculator import RewardCalculator

class MarketEnvironment(gym.Env):
    """Custom Market Environment for RL training"""
    
    def __init__(self,
                 data: pd.DataFrame,
                 action_space: ActionSpace,
                 market_state: MarketState,
                 reward_calculator: RewardCalculator):
        """Initialize market environment"""
        super(MarketEnvironment, self).__init__()
        
        self.data = data
        self.action_space = action_space
        self.market_state = market_state
        self.reward_calculator = reward_calculator
        self.current_step = 0
        
        # Define action and observation spaces for gym
        self.action_space = spaces.Discrete(action_space.n)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(market_state.state_features),),
            dtype=np.float32
        )
        
    def reset(self, seed=None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state"""
        super().reset(seed=seed)
        self.current_step = 0
        self.action_space.reset()
        return self._get_state(), {}
        
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Take action in environment"""
        # Execute action
        info = self.action_space.execute(
            action,
            self.data.iloc[self.current_step],
            self.data.iloc[self.current_step + 1]
        )
        
        # Calculate reward
        state = self._get_state()
        reward = self.reward_calculator.calculate_reward(action, state, info)
        
        # Move to next step
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        
        # Get next state
        next_state = self._get_state()
        
        return next_state, reward, done, False, info
        
    def _get_state(self) -> np.ndarray:
        """Get current state representation"""
        return self.market_state.get_state_features(
            self.data.iloc[:self.current_step + 1]
        )

    def render(self):
        """Render environment"""
        pass
