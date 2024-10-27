from typing import Dict
import numpy as np
import logging

class RewardCalculator:
    """Calculates rewards for RL actions"""
    
    def __init__(self):
        # Reward components and weights
        self.profit_weight = 0.4
        self.technical_weight = 0.3
        self.psychological_weight = 0.2
        self.risk_weight = 0.1
        
        # Thresholds
        self.profit_threshold = 0.02
        self.risk_threshold = 0.03
        
    def calculate_reward(self, action: int, state: np.ndarray, info: Dict) -> float:
        """Calculate composite reward for action"""
        try:
            # Calculate component rewards
            profit_reward = self._calculate_profit_reward(info)
            technical_reward = self._calculate_technical_reward(state, action)
            psychological_reward = self._calculate_psychological_reward(state)
            risk_penalty = self._calculate_risk_penalty(state, info)
            
            # Combine rewards
            total_reward = (
                profit_reward * self.profit_weight +
                technical_reward * self.technical_weight +
                psychological_reward * self.psychological_weight -
                risk_penalty * self.risk_weight
            )
            
            return float(np.clip(total_reward, -1, 1))
            
        except Exception as e:
            logging.error(f"Error calculating reward: {str(e)}")
            return 0.0

    def _calculate_profit_reward(self, info: Dict) -> float:
        """Calculate reward based on trade profit"""
        if not info['trade_executed']:
            return 0.0
            
        profit_pct = info['trade_profit'] / info['entry_price']
        return np.tanh(profit_pct / self.profit_threshold)

    def _calculate_technical_reward(self, state: np.ndarray, action: int) -> float:
        """Calculate reward based on technical alignment"""
        trend_strength = state[0]  # Index from MarketState features
        trend_consistency = state[1]
        
        if action == 1:  # Buy
            return (trend_strength + trend_consistency) / 2
        elif action == 2:  # Sell
            return -(trend_strength + trend_consistency) / 2
        return 0.0

    def _calculate_psychological_reward(self, state: np.ndarray) -> float:
        """Calculate reward based on psychological factors"""
        try:
            # Get psychological features from state
            sentiment = state[7]  # psychological_sentiment index
            zone_strength = state[8]  # zone_strength index
            
            # Calculate psychological alignment
            psych_reward = (sentiment + zone_strength) / 2
            return float(np.clip(psych_reward, -1, 1))
            
        except Exception as e:
            logging.error(f"Error calculating psychological reward: {str(e)}")
            return 0.0

    def _calculate_risk_penalty(self, state: np.ndarray, info: Dict) -> float:
        """Calculate penalty for excessive risk"""
        try:
            # Get risk-related features
            volatility = state[6]  # volatility index
            drawdown = info.get('drawdown', 0)
            
            # Calculate risk penalty
            risk_penalty = max(
                0,
                (volatility - self.risk_threshold) +
                (drawdown / self.risk_threshold)
            )
            
            return float(np.clip(risk_penalty, 0, 1))
            
        except Exception as e:
            logging.error(f"Error calculating risk penalty: {str(e)}")
            return 0.0

    def _calculate_holding_penalty(self, state: np.ndarray) -> float:
        """Calculate penalty for holding position"""
        try:
            # Get relevant features
            trend_strength = abs(state[0])
            volatility = state[6]
            
            # Higher penalty for holding in volatile, weak trend conditions
            holding_penalty = (1 - trend_strength) * volatility
            return float(np.clip(holding_penalty, 0, 1))
            
        except Exception as e:
            logging.error(f"Error calculating holding penalty: {str(e)}")
            return 0.0
