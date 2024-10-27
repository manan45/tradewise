import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any, List, Optional

from app.core.ai.reinforcement.action_space import ActionSpace
from app.core.ai.reinforcement.market_state import MarketState
from app.core.ai.reinforcement.reward_calculator import RewardCalculator

class MarketEnvironment(gym.Env):
    """Custom Market Environment for RL training"""
    
    def __init__(self,
                 data: Optional[pd.DataFrame] = None,
                 action_space: Optional[ActionSpace] = None,
                 market_state: Optional[MarketState] = None,
                 reward_calculator: Optional[RewardCalculator] = None):
        """Initialize market environment"""
        super(MarketEnvironment, self).__init__()
        
        self.data = data if data is not None else pd.DataFrame()
        self.action_space_obj = action_space if action_space is not None else ActionSpace()
        self.market_state = market_state if market_state is not None else MarketState()
        self.reward_calculator = reward_calculator if reward_calculator is not None else RewardCalculator()
        self.current_step = 0
        
        # Get state size from market state features
        self.state_size = len(self.market_state.state_features)
        
        # Define action and observation spaces for gym
        self.action_space = spaces.Discrete(self.action_space_obj.n)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.state_size,),
            dtype=np.float32
        )
        
    @property
    def action_size(self) -> int:
        """Get size of action space"""
        return self.action_space_obj.n
        
    def reset(self, seed=None) -> np.ndarray:
        """Reset environment to initial state"""
        super().reset(seed=seed)
        self.current_step = 0
        self.action_space_obj.reset()
        # Return only the state array, not the empty dict
        return self._get_state()
        
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Take action in environment"""
        try:
            # Execute action
            info = self.action_space_obj.execute(
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
            
        except Exception as e:
            self.logger.error(f"Error in step: {str(e)}")
            # Return safe defaults in case of error
            return self._get_state(), 0.0, True, False, {}
        
    def _get_state(self) -> np.ndarray:
        """Get current state representation"""
        try:
            state = self.market_state.get_state_features(
                self.data.iloc[:self.current_step + 1]
            )
            # Ensure state is a numpy array with correct shape
            return np.array(state, dtype=np.float32).reshape(self.state_size,)
        except Exception as e:
            self.logger.error(f"Error getting state: {str(e)}")
            # Return zero state in case of error
            return np.zeros(self.state_size, dtype=np.float32)

    def render(self):
        """Render environment"""
        pass
