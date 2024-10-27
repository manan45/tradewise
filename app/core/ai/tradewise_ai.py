import os
import traceback
from typing import Any, Dict, List, Tuple
from dotenv import load_dotenv
import threading
import asyncio
import pandas as pd
import numpy as np
from sqlalchemy import text
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from ta.trend import MACD, EMAIndicator, SMAIndicator, ADXIndicator
from ta.momentum import RSIIndicator, StochRSIIndicator, WilliamsRIndicator
from ta.volatility import BollingerBands, AverageTrueRange, KeltnerChannel
from ta.volume import AccDistIndexIndicator, OnBalanceVolumeIndicator, ForceIndexIndicator
from ta.others import DailyReturnIndicator, CumulativeReturnIndicator   
from stable_baselines3 import PPO
import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, LSTM, Dropout, Input
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
from decimal import Decimal, getcontext
import logging
import gymnasium
import gymnasium.spaces as spaces
from app.connectors.postgres_client import postgres_client
from app.core.repositories.stock_repository import StockRepository
from app.core.domain.models.trade_suggestion import DetailedTradeSuggestion
from scipy.stats import norm
import torch
import warnings

from .technical_indicators import TechnicalIndicatorCalculator
from .market_environment import MarketEnvironment
from .price_zone_analyzer import PriceZoneAnalyzer
from .market_psychology import MarketPsychology
from .trading_psychology import TradingPsychology
from .model_builder import build_lstm_model

# Set the precision for Decimal calculations
getcontext().prec = 10

load_dotenv()

class TradewiseAI:
    """Main trading system class"""
    
    def __init__(self, 
                 lookback_period: int = 100,
                 target_hours: int = 2,
                 conf_threshold: float = 0.7,
                 risk_reward_min: float = 2.0,
                 model_path: str = "./models/"):
        """Initialize TradewiseAI with configuration parameters"""
        self.lookback_period = lookback_period
        self.target_hours = target_hours
        self.conf_threshold = Decimal(str(conf_threshold))
        self.risk_reward_min = Decimal(str(risk_reward_min))
        self.model_path = model_path
        
        self.features = [
            'close', 'bb_middle', 'bb_upper', 'bb_lower', 
            'rsi', 'macd', 'volatility', 'bb_position', 'atr',
            'price_momentum', 'volume_momentum', 'support_strength',
            'resistance_strength', 'fib_ratio', 'price_in_range'
        ]
        self.scaler = MinMaxScaler()
        self.lstm_model = None
        self.rl_model = None
        self.stock_repository = StockRepository()
        self.indicator_calculator = TechnicalIndicatorCalculator()
        self.zone_analyzer = PriceZoneAnalyzer()
        self.market_psychology = MarketPsychology()
        self.trading_psychology = TradingPsychology()
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    async def generate_trade_suggestions(self, symbol, start_date: datetime, end_date: datetime):
        """Generate trade suggestions with proper formatting"""
        try:
            stock = await self.stock_repository.get_stock_by_symbol(symbol)
            if not stock:
                raise ValueError(f"Stock with symbol {symbol} not found")
            
            price_history = await self.stock_repository.get_price_history(symbol, start_date, end_date)
            df = pd.DataFrame([vars(price) for price in price_history])
            
            training_data = df[:-120]
            self.train_models(training_data)
            
            current_data = df[-120:]
            
            sentiment_score = self.calculate_sentiment(current_data)
            forecast_data = self.generate_timeseries_forecast(current_data)
            
            # Format suggestions
            suggestions = []
            for i, forecast in enumerate(forecast_data['forecasts'], 1):
                suggestion = {
                    'id': i,
                    'timestamp': forecast['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                    'Action': forecast['signals']['action'],
                    'Summary': {
                        'Current Price': f"${forecast['price']['current']:.2f}",
                        'Predicted Price': f"${forecast['price']['predicted']:.2f}",
                        'Expected Change': f"{((forecast['price']['predicted'] / forecast['price']['current']) - 1) * 100:.2f}%",
                        'Confidence': f"{forecast['signals']['confidence'] * 100:.1f}%"
                    },
                    'Technical Analysis': {
                        'RSI': f"{forecast['indicators']['rsi']:.1f}",
                        'Signal Strength': f"{forecast['signals']['strength']:.2f}",
                        'Volatility': f"{forecast['price']['volatility']:.2f}",
                    },
                    'Volume Analysis': {
                        'Trend': forecast['volume']['trend'],
                        'Strength': f"{forecast['volume']['strength'] * 100:.1f}%"
                    },
                    'Support/Resistance': {
                        'Support': f"${forecast['price']['support']:.2f}",
                        'Resistance': f"${forecast['price']['resistance']:.2f}"
                    }
                }

                if sentiment_score > 0.5:
                    suggestion['Action'] = 'BUY'
                elif sentiment_score < -0.5:
                    suggestion['Action'] = 'SELL'
                suggestions.append(suggestion)
            
            return suggestions
            
        except Exception as e:
            self.logger.error(f"Error in generate_trade_suggestions: {str(e)}")
            raise

    def generate_timeseries_forecast(self, current_data: pd.DataFrame) -> Dict:
        """Generate forecasts with psychological factors"""
        try:
            # Get key zones and psychology metrics
            zones = self.zone_analyzer.identify_key_zones(current_data)
            psychology = self.market_psychology.calculate_market_psychology(current_data, zones)
            
            # Prepare forecasts
            intervals = 24
            forecasts = []
            
            current_price = float(current_data['close'].iloc[-1])
            base_volatility = float(current_data['volatility'].mean())
            
            for interval in range(intervals):
                try:
                    timestamp = datetime.now() + timedelta(minutes=5 * (interval + 1))
                    
                    # Adjust prediction confidence based on psychology
                    confidence_modifier = (
                        psychology['trend_confidence'] * 0.3 +
                        psychology['support_confidence'] * 0.2 +
                        psychology['resistance_confidence'] * 0.2 +
                        (1 - abs(50 - psychology['fear_greed_index'])/50) * 0.3
                    )
                    
                    # Determine price targets based on zones
                    if psychology['breakout_probability'] > 0.7:
                        # Potential breakout scenario
                        if current_price > max(zones['support_zones']):
                            target_zone = min([z for z in zones['resistance_zones'] if z > current_price])
                            pred_price = current_price + (target_zone - current_price) * confidence_modifier
                        else:
                            target_zone = max([z for z in zones['support_zones'] if z < current_price])
                            pred_price = current_price - (current_price - target_zone) * confidence_modifier
                    elif psychology['reversal_probability'] > 0.7:
                        # Potential reversal scenario
                        if psychology['fear_greed_index'] > 70:
                            nearest_support = max([s for s in zones['support_zones'] if s < current_price])
                            pred_price = current_price - (current_price - nearest_support) * confidence_modifier
                        else:
                            nearest_resistance = min([r for r in zones['resistance_zones'] if r > current_price])
                            pred_price = current_price + (nearest_resistance - current_price) * confidence_modifier
                    else:
                        # Normal scenario
                        pred_price = self._generate_base_prediction(current_data, interval)
                    
                    # Add forecast to list
                    forecast = self._create_forecast_entry(current_price, pred_price, timestamp, psychology, zones)
                    forecasts.append(forecast)
                    
                    # Update current price for next iteration
                    current_price = pred_price
                    
                except Exception as e:
                    self.logger.warning(f"Error in interval {interval}: {str(e)}")
                    continue
            
            return {
                'forecasts': forecasts,
                'sequences': {
                    'timestamps': [(datetime.now() + timedelta(minutes=5*i)).strftime('%H:%M') 
                                   for i in range(len(forecasts)+1)],
                    'prices': [current_data['close'].iloc[-1]] + [f['price']['predicted'] for f in forecasts],
                    'volumes': [current_data['volume'].iloc[-1]] + [f['volume']['predicted'] for f in forecasts]
                }
            }
                
        except Exception as e:
            self.logger.error(f"Error in generate_timeseries_forecast: {str(e)}")
            raise

    def _generate_base_prediction(self, current_data: pd.DataFrame, interval: int) -> float:
        # Implement your base prediction logic here
        # This could involve using your LSTM model or other prediction methods
        pass

    def _create_forecast_entry(self, current_price: float, pred_price: float, 
                               timestamp: datetime, psychology: Dict, zones: Dict) -> Dict:
        # Implement the logic to create a forecast entry
        pass

    def train_models(self, data: pd.DataFrame):
        """Train LSTM and RL models"""
        try:
            # Preprocess data
            data = self.indicator_calculator.calculate_indicators(data)
            data = data.dropna()
            
            # Split data
            train_data, test_data = train_test_split(data, test_size=0.2, shuffle=False)
            
            # Train LSTM model
            self.lstm_model = build_lstm_model(train_data, self.features)
            self.lstm_model.fit(train_data[self.features], train_data['close'], epochs=10, batch_size=32)
            
            # Train RL model
            env = MarketEnvironment(data)
            self.rl_model = PPO("MlpPolicy", env, verbose=1)
            self.rl_model.learn(total_timesteps=10000)
            
            self.logger.info("Models trained successfully")
            
        except Exception as e:
            self.logger.error(f"Error training models: {str(e)}")
            raise

    def calculate_sentiment(self, data: pd.DataFrame) -> float:
        """Calculate sentiment score based on market psychology"""
        try:
            # Implement your sentiment calculation logic here
            # This could involve using the MarketPsychology class or other methods
            pass
        except Exception as e:
            self.logger.error(f"Error calculating sentiment: {str(e)}")
            raise

async def start_training(symbol):
    """Start training with proper error handling and output formatting"""
    ai = TradewiseAI()
    try:
        suggestions = await ai.generate_trade_suggestions(symbol)
        print(f"\nGenerated {len(suggestions)} forecasts for {symbol}")
        
        for suggestion in suggestions:
            print(f"\nForecast #{suggestion['id']} - {suggestion['timestamp']}")
            print(f"Recommended Action: {suggestion['Action']}")
            
            print("\nSummary:")
            for key, value in suggestion['Summary'].items():
                print(f"  {key}: {value}")
                
            print("\nTechnical Analysis:")
            for key, value in suggestion['Technical Analysis'].items():
                print(f"  {key}: {value}")
                
            print("\nVolume Analysis:")
            for key, value in suggestion['Volume Analysis'].items():
                print(f"  {key}: {value}")
                
            print("\nSupport/Resistance Levels:")
            for key, value in suggestion['Support/Resistance'].items():
                print(f"  {key}: {value}")
            
            print("-" * 80)
            
    except Exception as e:
        print(f"An error occurred while generating trade suggestions: {str(e)}")
        traceback.print_exc()

import os
import traceback
from typing import Any, Dict, List, Tuple
from dotenv import load_dotenv
import threading
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from decimal import Decimal, getcontext
import logging
import gymnasium
import gymnasium.spaces as spaces
from app.connectors.postgres_client import postgres_client
from app.core.repositories.stock_repository import StockRepository
from app.core.domain.models.trade_suggestion import DetailedTradeSuggestion
from scipy.stats import norm
import torch
import warnings

from .technical_indicators import TechnicalIndicatorCalculator
from .market_environment import MarketEnvironment
from .price_zone_analyzer import PriceZoneAnalyzer
from .market_psychology import MarketPsychology
from .trading_psychology import TradingPsychology
from .model_builder import build_lstm_model

# Set the precision for Decimal calculations
getcontext().prec = 10

load_dotenv()

class TradewiseAI:
    """Main trading system class"""
    
    def __init__(self, 
                 lookback_period: int = 100,
                 target_hours: int = 2,
                 conf_threshold: float = 0.7,
                 risk_reward_min: float = 2.0,
                 model_path: str = "./models/"):
        """Initialize TradewiseAI with configuration parameters"""
        self.lookback_period = lookback_period
        self.target_hours = target_hours
        self.conf_threshold = Decimal(str(conf_threshold))
        self.risk_reward_min = Decimal(str(risk_reward_min))
        self.model_path = model_path
        
        self.features = [
            'close', 'bb_middle', 'bb_upper', 'bb_lower', 
            'rsi', 'macd', 'volatility', 'bb_position', 'atr',
            'price_momentum', 'volume_momentum', 'support_strength',
            'resistance_strength', 'fib_ratio', 'price_in_range'
        ]
        self.scaler = MinMaxScaler()
        self.lstm_model = None
        self.rl_model = None
        self.stock_repository = StockRepository()
        self.indicator_calculator = TechnicalIndicatorCalculator()
        self.zone_analyzer = PriceZoneAnalyzer()
        self.market_psychology = MarketPsychology()
        self.trading_psychology = TradingPsychology()
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    async def generate_trade_suggestions(self, symbol):
        """Generate trade suggestions with proper formatting"""
        try:
            stock = await self.stock_repository.get_stock_by_symbol(symbol)
            if not stock:
                raise ValueError(f"Stock with symbol {symbol} not found")
            
            price_history = await self.stock_repository.get_price_history(symbol)
            df = pd.DataFrame([vars(price) for price in price_history])
            
            training_data = df[:-120]
            self.train_models(training_data)
            
            current_data = df[-120:]
            
            sentiment_score = self.calculate_sentiment(current_data)
            forecast_data = self.generate_timeseries_forecast(current_data)
            
            # Format suggestions
            suggestions = []
            for i, forecast in enumerate(forecast_data['forecasts'], 1):
                suggestion = {
                    'id': i,
                    'timestamp': forecast['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                    'Action': forecast['signals']['action'],
                    'Summary': {
                        'Current Price': f"${forecast['price']['current']:.2f}",
                        'Predicted Price': f"${forecast['price']['predicted']:.2f}",
                        'Expected Change': f"{((forecast['price']['predicted'] / forecast['price']['current']) - 1) * 100:.2f}%",
                        'Confidence': f"{forecast['signals']['confidence'] * 100:.1f}%"
                    },
                    'Technical Analysis': {
                        'RSI': f"{forecast['indicators']['rsi']:.1f}",
                        'Signal Strength': f"{forecast['signals']['strength']:.2f}",
                        'Volatility': f"{forecast['price']['volatility']:.2f}",
                    },
                    'Volume Analysis': {
                        'Trend': forecast['volume']['trend'],
                        'Strength': f"{forecast['volume']['strength'] * 100:.1f}%"
                    },
                    'Support/Resistance': {
                        'Support': f"${forecast['price']['support']:.2f}",
                        'Resistance': f"${forecast['price']['resistance']:.2f}"
                    }
                }

                if sentiment_score > 0.5:
                    suggestion['Action'] = 'BUY'
                elif sentiment_score < -0.5:
                    suggestion['Action'] = 'SELL'
                suggestions.append(suggestion)
            
            return suggestions
            
        except Exception as e:
            self.logger.error(f"Error in generate_trade_suggestions: {str(e)}")
            raise

    def generate_timeseries_forecast(self, current_data: pd.DataFrame) -> Dict:
        """Generate forecasts with psychological factors"""
        try:
            # Get key zones and psychology metrics
            zones = self.zone_analyzer.identify_key_zones(current_data)
            psychology = self.market_psychology.calculate_market_psychology(current_data, zones)
            
            # Prepare forecasts
            intervals = 24
            forecasts = []
            
            current_price = float(current_data['close'].iloc[-1])
            base_volatility = float(current_data['volatility'].mean())
            
            for interval in range(intervals):
                try:
                    timestamp = datetime.now() + timedelta(minutes=5 * (interval + 1))
                    
                    # Adjust prediction confidence based on psychology
                    confidence_modifier = (
                        psychology['trend_confidence'] * 0.3 +
                        psychology['support_confidence'] * 0.2 +
                        psychology['resistance_confidence'] * 0.2 +
                        (1 - abs(50 - psychology['fear_greed_index'])/50) * 0.3
                    )
                    
                    # Determine price targets based on zones
                    if psychology['breakout_probability'] > 0.7:
                        # Potential breakout scenario
                        if current_price > max(zones['support_zones']):
                            target_zone = min([z for z in zones['resistance_zones'] if z > current_price])
                            pred_price = current_price + (target_zone - current_price) * confidence_modifier
                        else:
                            target_zone = max([z for z in zones['support_zones'] if z < current_price])
                            pred_price = current_price - (current_price - target_zone) * confidence_modifier
                    elif psychology['reversal_probability'] > 0.7:
                        # Potential reversal scenario
                        if psychology['fear_greed_index'] > 70:
                            nearest_support = max([s for s in zones['support_zones'] if s < current_price])
                            pred_price = current_price - (current_price - nearest_support) * confidence_modifier
                        else:
                            nearest_resistance = min([r for r in zones['resistance_zones'] if r > current_price])
                            pred_price = current_price + (nearest_resistance - current_price) * confidence_modifier
                    else:
                        # Normal scenario
                        pred_price = self._generate_base_prediction(current_data, interval)
                    
                    # Add forecast to list
                    forecast = self._create_forecast_entry(current_price, pred_price, timestamp, psychology, zones)
                    forecasts.append(forecast)
                    
                    # Update current price for next iteration
                    current_price = pred_price
                    
                except Exception as e:
                    self.logger.warning(f"Error in interval {interval}: {str(e)}")
                    continue
            
            return {
                'forecasts': forecasts,
                'sequences': {
                    'timestamps': [(datetime.now() + timedelta(minutes=5*i)).strftime('%H:%M') 
                                   for i in range(len(forecasts)+1)],
                    'prices': [current_data['close'].iloc[-1]] + [f['price']['predicted'] for f in forecasts],
                    'volumes': [current_data['volume'].iloc[-1]] + [f['volume']['predicted'] for f in forecasts]
                }
            }
                
        except Exception as e:
            self.logger.error(f"Error in generate_timeseries_forecast: {str(e)}")
            raise

    def _generate_base_prediction(self, current_data: pd.DataFrame, interval: int) -> float:
        # Implement your base prediction logic here
        # This could involve using your LSTM model or other prediction methods
        pass

    def _create_forecast_entry(self, current_price: float, pred_price: float, 
                               timestamp: datetime, psychology: Dict, zones: Dict) -> Dict:
        # Implement the logic to create a forecast entry
        pass

    def train_models(self, data: pd.DataFrame):
        """Train LSTM and RL models"""
        try:
            # Preprocess data
            data = self.indicator_calculator.calculate_indicators(data)
            data = data.dropna()
            
            # Split data
            train_data, test_data = train_test_split(data, test_size=0.2, shuffle=False)
            
            # Train LSTM model
            self.lstm_model = build_lstm_model(train_data, self.features)
            self.lstm_model.fit(train_data[self.features], train_data['close'], epochs=10, batch_size=32)
            
            # Train RL model
            env = MarketEnvironment(data)
            self.rl_model = PPO("MlpPolicy", env, verbose=1)
            self.rl_model.learn(total_timesteps=10000)
            
            self.logger.info("Models trained successfully")
            
        except Exception as e:
            self.logger.error(f"Error training models: {str(e)}")
            raise

    def calculate_sentiment(self, data: pd.DataFrame) -> float:
        """Calculate sentiment score based on market psychology"""
        try:
            # Implement your sentiment calculation logic here
            # This could involve using the MarketPsychology class or other methods
            pass
        except Exception as e:
            self.logger.error(f"Error calculating sentiment: {str(e)}")
            raise

async def start_training(symbol):
    """Start training with proper error handling and output formatting"""
    ai = TradewiseAI()
    try:
        suggestions = await ai.generate_trade_suggestions(symbol)
        print(f"\nGenerated {len(suggestions)} forecasts for {symbol}")
        
        for suggestion in suggestions:
            print(f"\nForecast #{suggestion['id']} - {suggestion['timestamp']}")
            print(f"Recommended Action: {suggestion['Action']}")
            
            print("\nSummary:")
            for key, value in suggestion['Summary'].items():
                print(f"  {key}: {value}")
                
            print("\nTechnical Analysis:")
            for key, value in suggestion['Technical Analysis'].items():
                print(f"  {key}: {value}")
                
            print("\nVolume Analysis:")
            for key, value in suggestion['Volume Analysis'].items():
                print(f"  {key}: {value}")
                
            print("\nSupport/Resistance Levels:")
            for key, value in suggestion['Support/Resistance'].items():
                print(f"  {key}: {value}")
            
            print("-" * 80)
            
    except Exception as e:
        print(f"An error occurred while generating trade suggestions: {str(e)}")
        traceback.print_exc()
        logging.error(f"Error calculating psychological biases: {str(e)}")
        return {}

    def _calculate_anchoring_bias(self, df: pd.DataFrame) -> float:
        """Calculate susceptibility to anchoring bias"""
        try:
            # Compare current price to recent significant levels
            current_price = df['close'].iloc[-1]
            recent_high = df['high'].tail(20).max()
            recent_low = df['low'].tail(20).min()
            
            # Calculate price deviation from recent significant levels
            high_deviation = abs(current_price - recent_high) / recent_high
            low_deviation = abs(current_price - recent_low) / recent_low
            
            # Higher score indicates stronger anchoring to recent levels
            anchoring_score = 1 - min(high_deviation, low_deviation)
            
            return float(np.clip(anchoring_score, 0, 1))
            
        except Exception as e:
            logging.error(f"Error calculating anchoring bias: {str(e)}")
            return 0.5

    def get_psychological_advice(self, psychological_state: Dict[str, float]) -> Dict[str, str]:
        """Generate trading advice based on psychological state"""
        advice = {
            'risk_management': self._get_risk_advice(psychological_state),
            'emotional_control': self._get_emotional_advice(psychological_state),
            'bias_mitigation': self._get_bias_mitigation_advice(psychological_state),
            'confidence_adjustment': self._get_confidence_advice(psychological_state)
        }
        
        if psychological_state.get('in_support_zone') or psychological_state.get('in_resistance_zone'):
            advice['zone_management'] = self._get_zone_advice(psychological_state)
            
        return advice

    def _get_risk_advice(self, state: Dict[str, float]) -> str:
        """Generate risk management advice"""
        risk_tolerance = state.get('risk_tolerance', 0.5)
        stress_level = state.get('stress_level', 0.5)
        
        if risk_tolerance < 0.3:
            return "Consider reducing position sizes and implementing strict stop losses"
        elif risk_tolerance > 0.7 and stress_level > 0.6:
            return "Be cautious of overconfidence. Verify stop losses and position sizing."
        else:
            return "Maintain current risk management strategy but stay alert"

    def _get_emotional_advice(self, state: Dict[str, float]) -> str:
        """Generate emotional control advice"""
        emotional_balance = state.get('emotional_balance', 0.5)
        
        if emotional_balance < 0.3:
            return "High fear detected. Take a break to regain emotional balance."
        elif emotional_balance > 0.7:
            return "High greed detected. Review trades objectively before entering."
        else:
            return "Emotional state is balanced. Maintain this mindset."

    def _get_zone_advice(self, state: Dict[str, float]) -> str:
        """Generate advice for trading in support/resistance zones"""
        if state.get('in_support_zone'):
            return f"In support zone (strength: {state['zone_strength']:.2f}). Watch for reversal confirmation."
        elif state.get('in_resistance_zone'):
            return f"In resistance zone (strength: {state['zone_strength']:.2f}). Prepare for potential rejection."
        else:
            return "Outside key zones. Monitor price action for zone entries."

    def adjust_psychological_state(self, current_state: Dict[str, float], 
                                 market_conditions: Dict[str, float]) -> Dict[str, float]:
        """Dynamically adjust psychological state based on market conditions"""
        try:
            # Extract market conditions
            volatility = market_conditions.get('volatility', 0.5)
            trend_strength = market_conditions.get('trend_strength', 0.5)
            zone_pressure = market_conditions.get('zone_pressure', 0)
            
            # Adjust psychological metrics
            adjusted_state = current_state.copy()
            
            # Risk tolerance adjustment
            adjusted_state['risk_tolerance'] = self._adjust_risk_tolerance(
                current_state['risk_tolerance'],
                volatility,
                trend_strength
            )
            
            # Confidence adjustment
            adjusted_state['confidence'] = self._adjust_confidence(
                current_state['confidence'],
                zone_pressure,
                trend_strength
            )
            
            # Emotional balance adjustment
            adjusted_state['emotional_balance'] = self._adjust_emotional_balance(
                current_state['emotional_balance'],
                volatility,
                zone_pressure
            )
            
            return adjusted_state
            
        except Exception as e:
            logging.error(f"Error adjusting psychological state: {str(e)}")
            return current_state


class TradewiseAI:
    def __init__(self):
        # Add to existing initialization
        self.trading_psychology = TradingPsychology()
    
    def generate_forecast(self, current_data: pd.DataFrame, hours_ahead: int = 5) -> List[Dict]:
        try:
            df_indicators = self.calculate_indicators(current_data)
            # Calculate zones
            zone_analyzer = PriceZoneAnalyzer()
            zones = zone_analyzer.identify_key_zones(current_data)
            
            # Analyze psychology
            psychological_state = self.trading_psychology.analyze_trader_psychology(
                current_data,
                zones
            )
            
            # Get psychological advice
            trading_advice = self.trading_psychology.get_psychological_advice(psychological_state)
            
            scenarios = []
            current_price = Decimal(str(current_data['close'].iloc[-1]))
            
            for hour in range(1, hours_ahead + 1):
                # Original forecast logic...
                
                # Adjust forecast based on psychology
                scenario = self._analyze_scenario(
                    current_price,
                    predicted_price,
                    latest_indicators,
                    rl_action
                )
                
                # Add psychological factors
                scenario.update({
                    'psychology': {
                        'state': psychological_state,
                        'advice': trading_advice,
                        'confidence_adjustment': self._adjust_forecast_confidence(
                            scenario,
                            psychological_state
                        )
                    }
                })
                
                scenarios.append(scenario)
                
                # Update psychological state for next interval
                psychological_state = self.trading_psychology.adjust_psychological_state(
                    psychological_state,
                    {
                        '
