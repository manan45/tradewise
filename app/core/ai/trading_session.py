from dataclasses import dataclass
from typing import List, Dict, Union
import numpy as np
import pandas as pd
from datetime import datetime
import logging

@dataclass
class SessionState:
    confidence: float
    risk_tolerance: float
    emotional_balance: float
    stress_level: float
    market_sentiment: float
    zone_pressure: float

class TradingSession:
    """Manages trading sessions with psychological and technical analysis"""
    
    def __init__(self, 
                 initial_balance: float = 100000,
                 risk_per_trade: float = 0.02,
                 session_duration: int = 24):  # 2 hours in 5-min intervals
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.risk_per_trade = risk_per_trade
        self.session_duration = session_duration
        self.current_interval = 0
        self.final_balance = None  # Add this line to track final balance
        
        # Session states
        self.psychological_state = {}
        self.technical_state = {}
        self.zone_state = {}
        self.position = None
        
        # Session history
        self.trades = []
        self.state_history = []
        self.performance_metrics = {}
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
    def initialize_session(self, 
                         initial_psychology: Dict, 
                         market_data: pd.DataFrame,
                         zones: Union[Dict, List]):
        """Initialize a new trading session"""
        self.psychological_state = initial_psychology  # Changed from psychology to initial_psychology
        self.technical_state = self._analyze_technical_state(market_data)
        
        # Convert zones list to dict if needed
        if isinstance(zones, list):
            zones = {'levels': zones}
            
        self.zone_state = self._analyze_zone_state(market_data, zones)
        self.current_interval = 0
        self.balance = self.initial_balance
        self.position = None
        self.final_balance = None
        
        # Record initial state
        self._record_state()
        
    def _analyze_technical_state(self, df: pd.DataFrame) -> Dict:
        """Analyze current technical state"""
        latest = df.iloc[-1]
        
        return {
            'trend': {
                'direction': 'up' if latest['close'] > latest['sma_20'] else 'down',
                'strength': float(latest['trend_strength']),
                'momentum': float(latest['price_momentum'])
            },
            'oscillators': {
                'rsi': float(latest['rsi']),
                'macd': float(latest['macd']),
                'bb_position': float(latest['bb_position'])
            },
            'volatility': {
                'value': float(latest['volatility']),
                'atr': float(latest['atr']),
                'bb_width': float(latest['bb_width'])
            }
        }
    

    
    def update_session(self, 
                      new_price: float,
                      market_data: pd.DataFrame,
                      zones: Union[Dict, List],
                      prediction: Dict) -> Dict:
        """Update session state with new market data"""
        self.current_interval += 1
        
        # Update states
        self._update_psychological_state(new_price, prediction)
        self.technical_state = self._analyze_technical_state(market_data)
        
        # Convert zones list to dict if needed
        if isinstance(zones, list):
            zones = {'levels': zones}
            
        self.zone_state = self._analyze_zone_state(market_data, zones)
        
        # Process any open positions
        if self.position is not None:
            self._process_position(new_price)
            
        # Update final balance if session is ending
        if self.current_interval >= self.session_duration:
            self.final_balance = self.balance
        
        # Record updated state
        self._record_state()
        
        # Generate session insights
        return self._generate_session_insights()
    
    def _update_psychological_state(self, new_price: float, prediction: Dict):
        """Update psychological state based on price action and prediction"""
        # Update confidence based on prediction accuracy
        if self.psychological_state.get('last_prediction'):
            pred_accuracy = 1 - abs(
                (new_price - self.psychological_state['last_prediction']) / 
                self.psychological_state['last_prediction']
            )
            self.psychological_state['confidence'] = (
                self.psychological_state['confidence'] * 0.7 +
                pred_accuracy * 0.3
            )
        
        # Update emotional state based on position performance
        if self.position:
            pnl = (new_price - self.position['entry_price']) / self.position['entry_price']
            emotion_impact = np.clip(pnl * 2, -0.3, 0.3)  # Limit emotional impact
            self.psychological_state['emotional_balance'] = np.clip(
                self.psychological_state['emotional_balance'] + emotion_impact,
                0, 1
            )
        
        # Store current prediction
        self.psychological_state['last_prediction'] = prediction.get('predicted_price', new_price)
        
    def _process_position(self, current_price: float):
        """Process open position and update performance"""
        if not self.position:
            return
            
        # Calculate current P&L
        pnl = (current_price - self.position['entry_price']) / self.position['entry_price']
        
        # Check stop loss and take profit
        if (pnl <= self.position['stop_loss'] or 
            pnl >= self.position['take_profit'] or 
            self.current_interval >= self.session_duration):
            
            # Close position
            self.balance *= (1 + pnl)
            self.trades.append({
                'entry_price': self.position['entry_price'],
                'exit_price': current_price,
                'pnl': pnl,
                'duration': self.current_interval - self.position['entry_interval']
            })
            self.position = None
            
  
    def _generate_session_insights(self) -> Dict:
        """Generate insights from current session state"""
        return {
            'session_state': {
                'interval': self.current_interval,
                'psychological': {
                    'confidence': self.psychological_state.get('confidence', 0.5),
                    'emotional_balance': self.psychological_state.get('emotional_balance', 0.5)
                },
                'technical': {
                    'trend_direction': self.technical_state['trend']['direction'],
                    'trend_strength': self.technical_state['trend']['strength']
                },
                'zones': {
                    'in_support': self.zone_state['in_support_zone'],
                    'in_resistance': self.zone_state['in_resistance_zone']
                }
            },
            'trading_advice': self._generate_trading_advice(),
            'risk_advice': self._generate_risk_advice(),
            'performance': self._calculate_performance_metrics()
        }
    
    def _generate_trading_advice(self) -> Dict:
        """Generate trading advice based on current state"""
        confidence = self.psychological_state.get('confidence', 0.5)
        emotional_balance = self.psychological_state.get('emotional_balance', 0.5)
        trend_strength = self.technical_state['trend']['strength']
        
        # Base state assessment
        state_assessment = {
            'confidence': 'high' if confidence > 0.7 else 'low' if confidence < 0.3 else 'moderate',
            'emotional': 'balanced' if 0.4 <= emotional_balance <= 0.6 else 'extreme',
            'trend': 'strong' if trend_strength > 0.7 else 'weak' if trend_strength < 0.3 else 'moderate'
        }
        
        # Generate advice
        advice = {
            'entry_conditions': self._get_entry_conditions(),
            'position_sizing': self._get_position_sizing_advice(),
            'psychological_adjustment': self._get_psychological_adjustment()
        }
        
        return {
            'state_assessment': state_assessment,
            'advice': advice
        }
    
    def _get_entry_conditions(self) -> Dict:
        """Determine optimal entry conditions"""
        trend = self.technical_state['trend']
        zones = self.zone_state
        
        conditions = {
            'trend_alignment': trend['direction'],
            'momentum_confirmation': trend['momentum'] > 0,
            'zone_setup': 'support' if zones['in_support_zone'] else 
                         'resistance' if zones['in_resistance_zone'] else 'none',
            'entry_quality': min(
                self.psychological_state.get('confidence', 0.5),
                trend['strength'],
                zones.get('zone_strength', 0.5)
            )
        }
        
        return conditions
    
    def _get_position_sizing_advice(self) -> Dict:
        """Calculate optimal position size"""
        base_size = self.risk_per_trade
        
        # Adjust size based on confidence and market conditions
        confidence_factor = self.psychological_state.get('confidence', 0.5)
        trend_factor = self.technical_state['trend']['strength']
        volatility_factor = 1 - min(1, self.technical_state['volatility']['value'])
        
        adjusted_size = base_size * (
            confidence_factor * 0.4 +
            trend_factor * 0.3 +
            volatility_factor * 0.3
        )
        
        return {
            'base_size': base_size,
            'adjusted_size': adjusted_size,
            'adjustment_factors': {
                'confidence': confidence_factor,
                'trend': trend_factor,
                'volatility': volatility_factor
            }
        }
    
    def _get_psychological_adjustment(self) -> Dict:
        """Get psychological adjustment advice"""
        emotion = self.psychological_state.get('emotional_balance', 0.5)
        confidence = self.psychological_state.get('confidence', 0.5)
        
        adjustments = {
            'confidence_adjustment': 
                'increase' if confidence < 0.3 else
                'decrease' if confidence > 0.7 else
                'maintain',
            'emotional_adjustment':
                'calm' if emotion > 0.7 else
                'boost' if emotion < 0.3 else
                'maintain',
            'focus_points': []
        }
        
        # Add specific focus points
        if self.zone_state['in_support_zone'] or self.zone_state['in_resistance_zone']:
            adjustments['focus_points'].append('zone_confirmation')
        if abs(self.technical_state['trend']['momentum']) > 0.7:
            adjustments['focus_points'].append('momentum_control')
        if len(self.trades) > 0 and self.trades[-1]['pnl'] < 0:
            adjustments['focus_points'].append('recovery_mindset')
            
        return adjustments
    
    def _generate_risk_advice(self) -> Dict:
        """Generate risk management advice"""
        current_drawdown = (self.balance - self.initial_balance) / self.initial_balance
        recent_trades = self.trades[-3:] if len(self.trades) >= 3 else self.trades
        
        risk_state = {
            'drawdown': current_drawdown,
            'recent_performance': np.mean([t['pnl'] for t in recent_trades]) if recent_trades else 0,
            'volatility_risk': self.technical_state['volatility']['value'],
            'psychological_risk': 1 - self.psychological_state.get('emotional_balance', 0.5)
        }
        
        # Generate risk adjustments
        adjustments = {
            'position_size': 
                'reduce' if risk_state['drawdown'] < -0.05 or risk_state['psychological_risk'] > 0.7
                else 'increase' if all(t['pnl'] > 0 for t in recent_trades) and len(recent_trades) >= 3
                else 'maintain',
            'stop_distance':
                'widen' if risk_state['volatility_risk'] > 0.7
                else 'tighten' if risk_state['volatility_risk'] < 0.3
                else 'maintain',
            'trade_frequency':
                'decrease' if risk_state['psychological_risk'] > 0.6
                else 'increase' if risk_state['psychological_risk'] < 0.3
                else 'maintain'
        }
        
        return {
            'risk_state': risk_state,
            'adjustments': adjustments
        }
    
    def _calculate_performance_metrics(self) -> Dict:
        """Calculate session performance metrics"""
        if not self.trades:
            return {}
            
        pnls = [t['pnl'] for t in self.trades]
        
        return {
            'total_trades': len(self.trades),
            'win_rate': sum(1 for pnl in pnls if pnl > 0) / len(pnls),
            'avg_pnl': np.mean(pnls),
            'max_drawdown': min(0, min(pnls)),
            'profit_factor': abs(sum(pnl for pnl in pnls if pnl > 0) / 
                               sum(pnl for pnl in pnls if pnl < 0)) if any(pnl < 0 for pnl in pnls) else float('inf'),
            'psychological_stability': self.psychological_state.get('emotional_balance', 0.5)
        }
    
    def _analyze_zone_state(self, market_data: pd.DataFrame, zones: Union[Dict, List]) -> Dict:
        """Analyze the zone state based on market data and zones.
        
        Args:
            market_data: DataFrame containing market price data
            zones: Dictionary or List containing zone information
            
        Returns:
            Dictionary containing analyzed zone state
        """
        try:
            # Get current price safely
            current_price = float(market_data['close'].iloc[-1])
            
            # Initialize default zone state
            zone_state = {
                'current_price': current_price,
                'nearest_support': current_price * 0.9,
                'nearest_resistance': current_price * 1.1,
                'zone_strength': 0.5,
                'in_support_zone': False,
                'in_resistance_zone': False,
                'support_zones': [],
                'resistance_zones': []
            }
            
            # Convert list to dict if needed
            if isinstance(zones, list):
                zones = {'levels': zones}
            
            # Validate zones is now a dict
            if not isinstance(zones, dict):
                self.logger.warning(f"Invalid zones type after conversion: {type(zones)}, expected dict")
                return zone_state
                
            # Handle different possible zone dictionary structures
            if 'support_zones' in zones and 'resistance_zones' in zones:
                # Standard format
                support_zones = zones['support_zones']
                resistance_zones = zones['resistance_zones']
            elif 'levels' in zones:
                # Alternative format with combined levels
                support_zones = [level for level in zones['levels'] if level < current_price]
                resistance_zones = [level for level in zones['levels'] if level > current_price]
            elif 'zones' in zones:
                # Another possible format
                support_zones = [z['price'] for z in zones['zones'] if z.get('type') == 'support']
                resistance_zones = [z['price'] for z in zones['zones'] if z.get('type') == 'resistance']
            else:
                # Try to extract any numeric values as zones
                all_zones = [v for v in zones.values() if isinstance(v, (int, float))]
                support_zones = [z for z in all_zones if z < current_price]
                resistance_zones = [z for z in all_zones if z > current_price]
            
            # Ensure zones are lists of numbers
            support_zones = [float(z) for z in support_zones if isinstance(z, (int, float))]
            resistance_zones = [float(z) for z in resistance_zones if isinstance(z, (int, float))]
            
            # Calculate nearest zones
            if support_zones:
                zone_state['nearest_support'] = max([z for z in support_zones if z < current_price], 
                                                default=current_price * 0.9)
            
            if resistance_zones:
                zone_state['nearest_resistance'] = min([z for z in resistance_zones if z > current_price],
                                                    default=current_price * 1.1)
            
            # Calculate zone distances
            support_distance = (current_price - zone_state['nearest_support']) / current_price
            resistance_distance = (zone_state['nearest_resistance'] - current_price) / current_price
            
            # Define zone threshold (1% of price)
            zone_threshold = 0.01
            
            # Determine if price is in a zone
            zone_state['in_support_zone'] = support_distance <= zone_threshold
            zone_state['in_resistance_zone'] = resistance_distance <= zone_threshold
            
            # Calculate zone strength based on proximity and historical significance
            if zone_state['in_support_zone']:
                zone_state['zone_strength'] = 1 - (support_distance / zone_threshold)
            elif zone_state['in_resistance_zone']:
                zone_state['zone_strength'] = 1 - (resistance_distance / zone_threshold)
            else:
                zone_state['zone_strength'] = 0.0
            
            # Store available zones
            zone_state['support_zones'] = sorted(support_zones)
            zone_state['resistance_zones'] = sorted(resistance_zones)
            
            return zone_state
            
        except Exception as e:
            self.logger.error(f"Error analyzing zone state: {str(e)}\nZones data: {zones}")
            # Return safe default state
            return {
                'current_price': float(market_data['close'].iloc[-1]) if 'close' in market_data.columns else 0.0,
                'nearest_support': float(market_data['close'].iloc[-1] * 0.9) if 'close' in market_data.columns else 0.0,
                'nearest_resistance': float(market_data['close'].iloc[-1] * 1.1) if 'close' in market_data.columns else 0.0,
                'zone_strength': 0.0,
                'in_support_zone': False,
                'in_resistance_zone': False,
                'support_zones': [],
                'resistance_zones': []
            }

    def _record_state(self):
        """Record current session state with safe zone handling"""
        try:
            state = {
                'interval': self.current_interval,
                'psychological_state': self.psychological_state.copy(),
                'technical_state': self.technical_state.copy(),
                'zone_state': {
                    'current_price': self.zone_state.get('current_price', 0.0),
                    'nearest_support': self.zone_state.get('nearest_support', 0.0),
                    'nearest_resistance': self.zone_state.get('nearest_resistance', 0.0),
                    'zone_strength': self.zone_state.get('zone_strength', 0.0),
                    'in_support_zone': self.zone_state.get('in_support_zone', False),
                    'in_resistance_zone': self.zone_state.get('in_resistance_zone', False)
                },
                'balance': self.balance,
                'has_position': self.position is not None
            }
            self.state_history.append(state)
            
        except Exception as e:
            self.logger.error(f"Error recording state: {str(e)}")
            # Record minimal state if error occurs
            self.state_history.append({
                'interval': self.current_interval,
                'balance': self.balance,
                'has_position': self.position is not None
            })