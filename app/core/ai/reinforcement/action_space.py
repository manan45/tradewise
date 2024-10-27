from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import logging

class ActionSpace:
    """Defines and manages trading actions for RL"""
    
    def __init__(self):
        # Define action space
        self.n = 3  # hold, buy, sell
        self.actions = ['hold', 'buy', 'sell']
        
        # Position management
        self.max_position = 1.0
        self.min_position = -1.0
        self.position_size = 0.1
        self.current_position = 0.0
        
        # Risk management
        self.max_risk_per_trade = 0.02
        self.stop_loss_multiplier = 2.0
        
    def execute(self, action: int, current_data: pd.Series, next_data: pd.Series) -> Dict:
        """Execute trading action and return results"""
        try:
            action_type = self.actions[action]
            current_price = current_data['close']
            next_price = next_data['close']
            
            # Initialize info dictionary
            info = {
                'action': action_type,
                'entry_price': current_price,
                'exit_price': next_price,
                'position_delta': 0.0,
                'trade_executed': False,
                'trade_profit': 0.0,
                'stop_loss': 0.0,
                'take_profit': 0.0,
                'drawdown': 0.0
            }
            
            # Execute action based on type
            if action_type == 'buy':
                info.update(self._execute_buy(current_price, next_price))
            elif action_type == 'sell':
                info.update(self._execute_sell(current_price, next_price))
            
            # Calculate drawdown
            if self.current_position != 0:
                info['drawdown'] = self._calculate_drawdown(current_price, next_price)
            
            return info
            
        except Exception as e:
            logging.error(f"Error executing action: {str(e)}")
            return self._get_default_info()

    def _execute_buy(self, current_price: float, next_price: float) -> Dict:
        """Execute buy action"""
        if self.current_position >= self.max_position:
            return self._get_default_info()
            
        position_delta = min(
            self.position_size,
            self.max_position - self.current_position
        )
        
        self.current_position += position_delta
        
        stop_loss = current_price * (1 - self.max_risk_per_trade)
        take_profit = current_price * (1 + self.max_risk_per_trade * self.stop_loss_multiplier)
        
        return {
            'position_delta': position_delta,
            'trade_executed': True,
            'trade_profit': position_delta * (next_price - current_price),
            'stop_loss': stop_loss,
            'take_profit': take_profit
        }

    def _execute_sell(self, current_price: float, next_price: float) -> Dict:
        """Execute sell action"""
        if self.current_position <= self.min_position:
            return self._get_default_info()
            
        position_delta = min(
            self.position_size,
            self.current_position - self.min_position
        )
        
        self.current_position -= position_delta
        
        stop_loss = current_price * (1 + self.max_risk_per_trade)
        take_profit = current_price * (1 - self.max_risk_per_trade * self.stop_loss_multiplier)
        
        return {
            'position_delta': -position_delta,
            'trade_executed': True,
            'trade_profit': position_delta * (current_price - next_price),
            'stop_loss': stop_loss,
            'take_profit': take_profit
        }

    def _calculate_drawdown(self, current_price: float, next_price: float) -> float:
        """Calculate drawdown for current position"""
        if self.current_position > 0:
            return max(0, (current_price - next_price) / current_price)
        elif self.current_position < 0:
            return max(0, (next_price - current_price) / current_price)
        return 0.0

    def _get_default_info(self) -> Dict:
        """Get default info dictionary"""
        return {
            'position_delta': 0.0,
            'trade_executed': False,
            'trade_profit': 0.0,
            'stop_loss': 0.0,
            'take_profit': 0.0,
            'drawdown': 0.0
        }

    def reset(self):
        """Reset action space state"""
        self.current_position = 0.0
