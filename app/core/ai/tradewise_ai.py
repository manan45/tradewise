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
# Set the precision for Decimal calculations
getcontext().prec = 10

load_dotenv()

class TechnicalIndicatorCalculator:
    """Advanced technical indicator calculator with comprehensive market analysis"""
    
    def __init__(self):
        self.epsilon = 1e-8  # Small value to avoid division by zero

    def safe_divide(self, a, b):
        return np.divide(a, b, out=np.zeros_like(a), where=b!=0)
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate comprehensive technical indicators for market analysis
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with calculated technical indicators
        """
        df = df.copy()
        
        # Ensure all initial price/volume data is float type
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Replace infinite values with NaN
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        try:
            # === Price Action Indicators ===
            self._calculate_price_action_indicators(df)
            
            # === Trend Indicators ===
            self._calculate_trend_indicators(df)
            
            # === Momentum Indicators ===
            self._calculate_momentum_indicators(df)
            
            # === Volatility Indicators ===
            self._calculate_volatility_indicators(df)
            
            # === Volume Indicators ===
            self._calculate_volume_indicators(df)
            
            # === Support and Resistance ===
            self._calculate_support_resistance(df)
            
            # === Fibonacci Levels ===
            self._calculate_fibonacci_levels(df)
            
            # === Custom Composite Indicators ===
            self._calculate_composite_indicators(df)
            
            # Final data cleaning
            self._clean_dataframe(df)
            
            return df
            
        except Exception as e:
            raise ValueError(f"Error calculating indicators: {str(e)}")
    
    def _calculate_price_action_indicators(self, df: pd.DataFrame):
        """Calculate price action based indicators"""
        # Candlestick patterns
        df['body_size'] = abs(df['close'] - df['open'])
        df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
        df['body_ratio'] = df['body_size'] / (df['high'] - df['low'])
        
        # Price momentum
        df['price_momentum'] = df['close'].pct_change(5, fill_method=None)
        df['price_momentum'] = df['price_momentum'].fillna(0)
        df['price_acceleration'] = df['price_momentum'].diff()
        
        # Moving averages
        for period in [10, 20, 50, 200]:
            df[f'sma_{period}'] = SMAIndicator(close=df['close'], window=period).sma_indicator()
            df[f'ema_{period}'] = EMAIndicator(close=df['close'], window=period).ema_indicator()
        
        # Golden/Death Cross
        df['golden_cross'] = (df['sma_50'] > df['sma_200']).astype(int)
        df['death_cross'] = (df['sma_50'] < df['sma_200']).astype(int)
    
    def _calculate_trend_indicators(self, df: pd.DataFrame):
        """Calculate trend-following indicators"""
        # MACD
        macd = MACD(close=df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_hist'] = macd.macd_diff()
        
        # ADX
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            adx_indicator = ADXIndicator(high=df['high'], low=df['low'], close=df['close'])
            df['adx'] = adx_indicator.adx()
            df['di_plus'] = adx_indicator.adx_pos()
            df['di_minus'] = adx_indicator.adx_neg()

        # Handle potential division by zero and NaN values
        df['adx'] = df['adx'].replace([np.inf, -np.inf], np.nan).fillna(0)
        df['di_plus'] = df['di_plus'].replace([np.inf, -np.inf], np.nan).fillna(0)
        df['di_minus'] = df['di_minus'].replace([np.inf, -np.inf], np.nan).fillna(0)
        df['adx'] = self.safe_divide(df['adx'], df['adx'].abs().rolling(window=14).mean() + self.epsilon) * 100
        
        # Bollinger Bands
        bb = BollingerBands(close=df['close'])
        df['bb_middle'] = bb.bollinger_mavg()
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_width'] = self.safe_divide(df['bb_upper'] - df['bb_lower'], df['bb_middle'])
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    def _calculate_momentum_indicators(self, df: pd.DataFrame):
        """Calculate momentum-based indicators"""
        # RSI and Stochastic
        df['rsi'] = RSIIndicator(close=df['close']).rsi()
        stoch_rsi = StochRSIIndicator(close=df['close'])
        df['stoch_rsi_k'] = stoch_rsi.stochrsi_k()
        df['stoch_rsi_d'] = stoch_rsi.stochrsi_d()
        
        # Williams %R
        df['williams_r'] = WilliamsRIndicator(
            high=df['high'],
            low=df['low'],
            close=df['close']
        ).williams_r()
        
        # Rate of Change
        for period in [5, 10, 20]:
            df[f'roc_{period}'] = df['close'].pct_change(period, fill_method=None)
            df[f'roc_{period}'] = df[f'roc_{period}'].fillna(0)
    
    def _calculate_volatility_indicators(self, df: pd.DataFrame):
        """Calculate volatility indicators"""
        # ATR and Normalized ATR
        atr = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'])
        df['atr'] = atr.average_true_range()
        df['natr'] = df['atr'] / df['close']
        
        # Historical Volatility
        df['volatility'] = df['close'].rolling(window=20).std()
        df['volatility_ratio'] = self.safe_divide(df['volatility'], df['volatility'].rolling(window=100).mean())
        
        # Keltner Channels
        kc = KeltnerChannel(high=df['high'], low=df['low'], close=df['close'])
        df['kc_middle'] = kc.keltner_channel_mband()
        df['kc_upper'] = kc.keltner_channel_hband()
        df['kc_lower'] = kc.keltner_channel_lband()
    
    def _calculate_volume_indicators(self, df: pd.DataFrame):
        """Calculate volume-based indicators"""
        # On-Balance Volume
        df['obv'] = OnBalanceVolumeIndicator(close=df['close'], volume=df['volume']).on_balance_volume()
        
        # Accumulation/Distribution
        df['acc_dist'] = AccDistIndexIndicator(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            volume=df['volume']
        ).acc_dist_index()
        
        # Force Index
        df['force_index'] = ForceIndexIndicator(
            close=df['close'],
            volume=df['volume']
        ).force_index()
        
        # Volume momentum
        df['volume_momentum'] = df['volume'].pct_change(5, fill_method=None)
        df['volume_momentum'] = df['volume_momentum'].fillna(0)
        
        df['volume_sma_ratio'] = self.safe_divide(df['volume'], df['volume'].rolling(window=20).mean())
    
    def _calculate_support_resistance(self, df: pd.DataFrame):
        """Calculate support and resistance levels"""
        # Dynamic support and resistance
        window = 20
        df['support'] = df['low'].rolling(window=window).min()
        df['resistance'] = df['high'].rolling(window=window).max()
        
        # Strength indicators
        df['support_strength'] = self.safe_divide(df['close'] - df['support'], df['support'])
        df['resistance_strength'] = self.safe_divide(df['resistance'] - df['close'], df['close'])
        df['sr_range'] = df['resistance'] - df['support']
        df['price_in_range'] = self.safe_divide(df['close'] - df['support'], df['sr_range'])
    
    def _calculate_fibonacci_levels(self, df: pd.DataFrame):
        """Calculate Fibonacci retracement levels"""
        high = df['high'].rolling(window=20).max()
        low = df['low'].rolling(window=20).min()
        diff = high - low
        
        # Standard Fibonacci levels
        fib_levels = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1]
        for level in fib_levels:
            df[f'fib_{int(level*100)}'] = low + level * diff
        
        # Calculate distance to closest Fibonacci level
        def find_closest_fib(row):
            levels = [row[f'fib_{int(level*100)}'] for level in fib_levels]
            return min(levels, key=lambda x: abs(x - row['close']))
        
        df['closest_fib'] = df.apply(find_closest_fib, axis=1)
        df['fib_ratio'] = self.safe_divide(df['close'] - df['fib_0'], df['fib_100'] - df['fib_0'])
    
    def _calculate_composite_indicators(self, df: pd.DataFrame):
        """Calculate custom composite indicators"""
        # Trend strength composite
        df['trend_strength'] = (
            (df['close'] > df['sma_20']).astype(int) +
            (df['close'] > df['sma_50']).astype(int) +
            (df['close'] > df['sma_200']).astype(int) +
            (df['macd'] > 0).astype(int) +
            (df['rsi'] > 50).astype(int)
        ) / 5
        
        # Volatility composite
        df['volatility_score'] = (
            self.safe_divide(df['natr'], df['natr'].rolling(window=100).mean()) +
            self.safe_divide(df['bb_width'], df['bb_width'].rolling(window=100).mean())
        ) / 2
        
        # Volume quality
        df['volume_quality'] = (
            (df['volume'] > df['volume'].rolling(window=20).mean()).astype(int) *
            df['volume_momentum'].abs()
        )
        
        # Overall market strength
        df['market_strength'] = (
            df['trend_strength'] * 0.4 +
            df['rsi'] / 100 * 0.2 +
            df['volume_quality'] * 0.2 +
            (1 - df['volatility_score']) * 0.2
        )
    
    def _clean_dataframe(self, df: pd.DataFrame):
        """Clean and validate the dataframe"""
        # Handle NaN values
        df = df.ffill().fillna(0)
        
        # Ensure all columns are float type
        for col in df.columns:
            if col != 'closest_fib':  # Skip non-numeric columns
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Clip extreme values
        for col in df.columns:
            if col != 'closest_fib':
                mean = df[col].mean()
                std = df[col].std()
                df[col] = df[col].clip(mean - 4*std, mean + 4*std)



class MarketEnvironment(gymnasium.Env):
    def __init__(self, data: pd.DataFrame, initial_balance: float = 100000):
        super().__init__()
        # Ensure data is preprocessed and cleaned
        self.data = self._preprocess_data(data)
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.position = None
        self.current_step = 0
        self.max_steps = len(self.data) - 1
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(3)  # Buy, Sell, Hold
        observation_dim = 15  # Number of features
        
        # Set more reasonable bounds for observation space
        self.observation_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(observation_dim,),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        """Reset the environment"""
        super().reset(seed=seed)  # Initialize RNG state
        
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = None
        
        # Get first observation
        observation = self._get_observation()
        
        # Return observation and info dict
        info = {}
        return observation, info

    def step(self, action):
        """Execute one environment step"""
        # Ensure action is valid
        assert self.action_space.contains(action)
        
        # Get current price data
        current_price = float(self.data.iloc[self.current_step]['close'])
        
        # Default values
        reward = 0.0
        done = False
        
        # Execute trade action
        if action == 0:  # Buy
            if self.position is None:
                self.position = current_price
                reward = -0.0001  # Small transaction cost
        
        elif action == 1:  # Sell
            if self.position is not None:
                reward = (current_price - self.position) / self.position
                self.position = None
        
        # Increment step
        self.current_step += 1
        
        # Check if episode is done
        if self.current_step >= self.max_steps:
            done = True
            # Force sell if still holding at end of episode
            if self.position is not None:
                reward = (current_price - self.position) / self.position
                self.position = None
        
        # Get next observation
        observation = self._get_observation()
        
        # Clip reward to prevent extreme values
        reward = float(np.clip(reward, -1.0, 1.0))
        
        info = {
            'current_step': self.current_step,
            'current_price': current_price,
            'position': self.position,
            'balance': self.balance
        }
        
        return observation, reward, done, False, info  # False is for truncated flag in gymnasium

    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess data to ensure numerical stability"""
        df = df.copy()
        
        # Replace infinite values with NaN
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Forward fill then backward fill NaN values
        df = df.ffill().bfill()
        
        # Clip extreme values based on percentiles
        for column in df.select_dtypes(include=[np.number]).columns:
            q1 = df[column].quantile(0.01)
            q3 = df[column].quantile(0.99)
            df[column] = df[column].clip(q1, q3)
            
            # Normalize if values are too large
            if df[column].abs().max() > 10:
                df[column] = df[column] / df[column].abs().max() * 10
        
        return df

    def _get_observation(self):
        """Get normalized current market state"""
        if self.current_step >= len(self.data):
            return np.zeros(self.observation_space.shape, dtype=np.float32)
            
        features = [
            'close', 'bb_middle', 'bb_upper', 'bb_lower', 
            'rsi', 'macd', 'volatility', 'bb_position', 'atr',
            'price_momentum', 'volume_momentum', 'support_strength',
            'resistance_strength', 'fib_ratio', 'price_in_range'
        ]
        
        observation = []
        for feature in features:
            try:
                value = self.data.iloc[self.current_step][feature]
                if pd.isna(value) or np.isinf(value):
                    value = 0.0
                else:
                    value = float(value)
                    value = np.clip(value, -10, 10)
            except Exception:
                value = 0.0
            observation.append(value)
        
        obs_array = np.array(observation, dtype=np.float32)
        
        # Additional safety check
        if np.any(np.isnan(obs_array)) or np.any(np.isinf(obs_array)):
            obs_array = np.zeros_like(obs_array)
            
        return obs_array

    def render(self, mode='human'):
        """Render the environment"""
        pass

    def close(self):
        """Close the environment"""
        pass

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
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def calculate_sentiment(self, df: pd.DataFrame) -> float:
        """
        Calculate a sentiment score based on recent price movements or external data.
        
        Args:
            df: DataFrame with recent price data.
        
        Returns:
            A float representing the sentiment score.
        """
        # Example: Simple sentiment based on recent price change
        recent_change = df['close'].pct_change().tail(10).mean()
        sentiment_score = np.clip(recent_change * 100, -1, 1)  # Scale to [-1, 1]
        return sentiment_score

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for the given price data"""
        return self.indicator_calculator.calculate_indicators(df)

    def build_lstm_model(self, input_shape):
        inputs = Input(shape=input_shape)
        x = LSTM(128, return_sequences=True)(inputs)
        x = Dropout(0.3)(x)
        x = LSTM(64, return_sequences=True)(x)
        x = Dropout(0.2)(x)
        x = LSTM(32, return_sequences=False)(x)
        x = Dropout(0.2)(x)
        x = Dense(16, activation='relu')(x)
        outputs = Dense(1)(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='huber', metrics=['mae', 'mse'])
        return model

    def prepare_data(self, df: pd.DataFrame, target_hours: int = 2) -> Tuple[np.ndarray, np.ndarray]:
        df_indicators = self.calculate_indicators(df)
        
        # Ensure all features are present and numeric
        for feature in self.features:
            if feature not in df_indicators.columns:
                raise ValueError(f"Feature '{feature}' not found in calculated indicators")
            df_indicators[feature] = pd.to_numeric(df_indicators[feature], errors='coerce')
        
        # Replace any remaining NaN or inf values with 0
        df_indicators[self.features] = df_indicators[self.features].replace([np.inf, -np.inf], np.nan).fillna(0)
        
        scaled_data = self.scaler.fit_transform(df_indicators[self.features])
        
        X, y = [], []
        for i in range(self.lookback_period, len(scaled_data) - target_hours):
            X.append(scaled_data[i-self.lookback_period:i])
            y.append(scaled_data[i+target_hours, 0])  # Predicting close price
        
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    def train_models(self, df: pd.DataFrame) -> None:
        """Train both LSTM and RL models with improved stability"""
        try:
            X, y = self.prepare_data(df)
            
            # Check for NaN values after preprocessing
            if np.isnan(X).any() or np.isnan(y).any():
                raise ValueError("NaN values still present after preprocessing")
            
            # Train LSTM model
            self.lstm_model = self.build_lstm_model((X.shape[1], X.shape[2]))
            self.lstm_model.fit(X, y, epochs=50, batch_size=32, validation_split=0.1, verbose=0)
            
            # Create and train RL model with more stable configuration
            env = MarketEnvironment(df)
            
            # Fixed PPO initialization - Pass the activation function class, not the function call
            self.rl_model = PPO(
                'MlpPolicy',
                env,
                verbose=0,
                policy_kwargs={
                    'net_arch': [64, 64],
                    'activation_fn': torch.nn.ReLU,
                    'ortho_init': True
                },
                learning_rate=1e-4,
                n_steps=1024,
                batch_size=64,
                n_epochs=5,
                gamma=0.95,
                gae_lambda=0.9,
                clip_range=0.1,
                ent_coef=0.005,
                vf_coef=0.5,
                max_grad_norm=0.3,
                use_sde=False,
                normalize_advantage=True
            )
            
            # Train with fewer steps initially
            self.rl_model.learn(
                total_timesteps=50000,
                progress_bar=True
            )
            
            self.logger.info("Models trained successfully")
        
        except Exception as e:
            self.logger.error(f"Error during model training: {str(e)}")
            raise

    def generate_forecast(self, current_data: pd.DataFrame, hours_ahead: int = 5) -> List[Dict]:
        try:
            df_indicators = self.calculate_indicators(current_data)
            sentiment_score = self.calculate_sentiment(current_data)
            
            recent_data = df_indicators[self.features].tail(self.lookback_period)
            recent_data = recent_data.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0)
            
            if self.lstm_model is None or self.rl_model is None:
                raise ValueError("Models not trained. Call train_models first.")
            
            scenarios = []
            current_price = Decimal(str(current_data['close'].iloc[-1]))
            base_scaled_data = self.scaler.transform(recent_data)
            latest_indicators = df_indicators.iloc[-1]
            
            volatility = float(latest_indicators['volatility'])
            
            for hour in range(1, hours_ahead + 1):
                X = base_scaled_data.reshape(1, self.lookback_period, len(self.features))
                base_pred = self.lstm_model.predict(X, verbose=0)[0][0]
                
                # Create a dummy array with the same shape as the original features
                dummy_pred = np.zeros((1, len(self.features)))
                dummy_pred[0, 0] = base_pred  # Set the first feature (close price) to the predicted value
                
                # Inverse transform the dummy array
                price_pred = self.scaler.inverse_transform(dummy_pred)[0][0]
                
                noise = np.random.normal(0, volatility * 0.1)
                price_pred *= (1 + noise)
                
                state = self._get_state(latest_indicators)
                action, _ = self.rl_model.predict(state)
                
                # Adjust action based on sentiment
                if sentiment_score > 0.5:
                    action = 0  # Bias towards Buy
                elif sentiment_score < -0.5:
                    action = 1  # Bias towards Sell

                scenario = self._analyze_scenario(
                    current_price,
                    Decimal(str(price_pred)),
                    latest_indicators,
                    int(action)
                )
                scenario['hour'] = hour
                scenarios.append(scenario)
            
            return scenarios
        except Exception as e:
            self.logger.error(f"Error in generate_forecast: {str(e)}")
            raise

    def _update_indicators_sequence(self, indicators: pd.Series, 
                              price_sequence: List[float], 
                              volume_sequence: List[float], 
                              interval: int) -> pd.Series:
        """Update technical indicators based on predicted price and volume sequence"""
        try:
            updated = indicators.copy()
            
            # Convert sequences to numpy arrays and ensure float type
            price_sequence = np.array(price_sequence, dtype=np.float64)
            volume_sequence = np.array(volume_sequence, dtype=np.float64)
            
            # Ensure we have valid sequences
            if len(price_sequence) < 2 or len(volume_sequence) < 2:
                return indicators
                
            # Calculate returns safely
            returns = np.diff(price_sequence) / price_sequence[:-1]
            returns = np.nan_to_num(returns, 0)  # Replace NaN with 0
            
            # Basic indicator updates with type safety
            updated['close'] = float(price_sequence[-1])
            updated['volatility'] = float(np.std(returns)) if len(returns) > 1 else float(indicators['volatility'])
            updated['price_momentum'] = float(returns[-1]) if returns.size > 0 else 0.0
            
            # Volume momentum with type safety
            volume_change = 0.0
            if volume_sequence[-2] > 0:
                volume_change = float((volume_sequence[-1] - volume_sequence[-2]) / volume_sequence[-2])
            updated['volume_momentum'] = volume_change
            
            # Moving averages with type safety
            if len(price_sequence) >= 20:
                updated['sma_20'] = float(np.mean(price_sequence[-20:]))
            if len(price_sequence) >= 50:
                updated['sma_50'] = float(np.mean(price_sequence[-50:]))
            
            # RSI calculation with type safety
            gains = np.sum([x for x in returns if x > 0]) if returns.size > 0 else 0
            losses = abs(np.sum([x for x in returns if x < 0])) if returns.size > 0 else 0
            
            if losses != 0:
                rs = gains / losses
                updated['rsi'] = min(100.0, max(0.0, float(100.0 - (100.0 / (1.0 + rs)))))
            else:
                updated['rsi'] = 100.0 if gains > 0 else 50.0
            
            # Trend strength with type safety
            if returns.size > 0:
                look_back = min(6, len(returns))
                trend_signals = [1 if returns[-i] > 0 else 0 for i in range(1, look_back+1)]
                updated['trend_strength'] = float(np.mean(trend_signals))
            else:
                updated['trend_strength'] = 0.5
                
            # Convert all values to float explicitly
            for key in updated.index:
                try:
                    if isinstance(updated[key], (np.floating, np.integer)):
                        updated[key] = float(updated[key])
                    elif not isinstance(updated[key], float):
                        updated[key] = float(indicators[key])
                except:
                    updated[key] = 0.0
                    
            return updated
            
        except Exception as e:
            self.logger.warning(f"Error in _update_indicators_sequence: {str(e)}")
            return indicators

    
    def _adjust_rsi(self, current_rsi: float, price_change: float) -> float:
        """Adjust RSI based on price change"""
        # Simplified RSI adjustment
        if price_change > 0:
            return min(100, current_rsi + (price_change * 100))
        else:
            return max(0, current_rsi + (price_change * 100))

    def _get_state(self, row: pd.Series) -> np.ndarray:
        """Get current market state for RL model"""
        return np.array([
            row['close'], row['bb_middle'], row['bb_upper'], row['bb_lower'],
            row['rsi'], row['macd'], row['volatility'], row['bb_position'],
            row['atr'], row['price_momentum'], row['volume_momentum'],
            row['support_strength'], row['resistance_strength'],
            row['fib_ratio'], row['price_in_range']
        ])

    def _analyze_scenario(self, current_price: Decimal, predicted_price: Decimal, indicators: pd.Series, rl_action: int) -> Dict[str, Any]:
        """
        Analyze a trading scenario and generate detailed metrics.

        Args:
            current_price (Decimal): The current price of the asset.
            predicted_price (Decimal): The predicted price of the asset.
            indicators (pd.Series): A series containing technical indicators.
            rl_action (int): The action suggested by the reinforcement learning model.

        Returns:
            Dict[str, Any]: A dictionary containing detailed analysis of the scenario.
        """
        price_change = (predicted_price - current_price) / current_price

        # Calculate technical signals
        bb_signal = Decimal('1') if current_price < Decimal(str(indicators['bb_lower'])) else \
                    Decimal('-1') if current_price > Decimal(str(indicators['bb_upper'])) else Decimal('0')
        rsi_signal = Decimal('1') if Decimal(str(indicators['rsi'])) < Decimal('30') else \
                     Decimal('-1') if Decimal(str(indicators['rsi'])) > Decimal('70') else Decimal('0')
        macd_signal = Decimal('1') if Decimal(str(indicators['macd'])) > Decimal(str(indicators['macd_signal'])) else Decimal('-1')
        
        # Support and resistance signals
        support_signal = Decimal('1') if Decimal(str(indicators['support_strength'])) < Decimal('0.05') else Decimal('0')
        resistance_signal = Decimal('-1') if Decimal(str(indicators['resistance_strength'])) < Decimal('0.05') else Decimal('0')
        
        # Fibonacci signal
        fib_signal = Decimal('1') if Decimal(str(indicators['fib_ratio'])) < Decimal('0.382') else \
                     Decimal('-1') if Decimal(str(indicators['fib_ratio'])) > Decimal('0.618') else Decimal('0')
        
        signal_strength = (bb_signal + rsi_signal + macd_signal + support_signal + resistance_signal + fib_signal) / Decimal('6')
        
        volatility = Decimal(str(indicators['volatility']))
        atr = Decimal(str(indicators['atr']))
        
        # Calculate stop loss and take profit levels
        stop_loss = current_price * (Decimal('1') - Decimal('2') * atr / current_price)
        take_profit = current_price * (Decimal('1') + Decimal('3') * atr / current_price)
        
        # Calculate probability of profit (using float for norm.cdf)
        prob_profit = Decimal(str(1 - norm.cdf(float(current_price), float(predicted_price), float(volatility))))
        
        # Calculate expected return
        expected_return = (
            (take_profit - current_price) * prob_profit +
            (stop_loss - current_price) * (Decimal('1') - prob_profit)
        ) / current_price
        
        # Calculate risk-reward ratio
        risk_reward_ratio = abs((take_profit - current_price) / (current_price - stop_loss))
        
        # Determine suggested position size
        suggested_position_size = min(Decimal('0.1'), abs(signal_strength) * Decimal('0.1'))
        
        return {
            'timestamp': datetime.now() + timedelta(hours=2),
            'current_price': current_price,
            'predicted_price': predicted_price,
            'price_change': price_change,
            'signal_strength': signal_strength,
            'rl_action': ['BUY', 'SELL', 'HOLD'][rl_action],
            'confidence': abs(signal_strength),
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'probability_profit': prob_profit,
            'expected_return': expected_return,
            'risk_reward_ratio': risk_reward_ratio,
            'volatility': volatility,
            'bb_position': Decimal(str(indicators['bb_position'])),
            'rsi': Decimal(str(indicators['rsi'])),
            'suggested_position_size': suggested_position_size,
            'technical_indicators': {
                'bb_signal': bb_signal,
                'rsi_signal': rsi_signal,
                'macd_signal': macd_signal,
                'support_signal': support_signal,
                'resistance_signal': resistance_signal,
                'fib_signal': fib_signal
            },
            'support_strength': Decimal(str(indicators['support_strength'])),
            'resistance_strength': Decimal(str(indicators['resistance_strength'])),
            'fib_ratio': Decimal(str(indicators['fib_ratio'])),
            'price_in_range': Decimal(str(indicators['price_in_range'])),
            'support_and_resistance': {
                'support': indicators['support_strength'],
                'resistance': indicators['resistance_strength'],
                'price_in_range': indicators['price_in_range']
            }
        }
    
    def _format_time_series_suggestions(self, scenarios: List[Dict]) -> List[Dict]:
        """Format time series forecasts into structured suggestions"""
        suggestions = []
        
        for scenario in scenarios:
            suggestion = {
                'Suggestion': f"#{scenario['hour']}",
                'Action': scenario['rl_action'],
                'Summary': {
                    'Current Price': f"${scenario['current_price']:.2f}",
                    'Predicted Price': f"${scenario['predicted_price']:.2f}",
                    'Expected Return': f"{scenario['expected_return']*100:.2f}%",
                    'Confidence': f"{scenario['confidence']*100:.1f}%",
                    'Probability of Profit': f"{scenario['probability_profit']*100:.1f}%"
                },
                'Risk Management': {
                    'Stop Loss': f"${scenario['stop_loss']:.2f} ({(scenario['stop_loss']/scenario['current_price']-1)*100:.1f}%)",
                    'Take Profit': f"${scenario['take_profit']:.2f} ({(scenario['take_profit']/scenario['current_price']-1)*100:.1f}%)",
                    'Risk/Reward Ratio': f"{scenario['risk_reward_ratio']:.2f}",
                    'Suggested Position Size': f"{scenario['suggested_position_size']*100:.1f}% of portfolio"
                },
                'Technical Analysis': {
                    'RSI': f"{scenario['rsi']:.1f}",
                    'Bollinger Position': f"{scenario['bb_position']:.2f}",
                    'Volatility': f"{scenario['volatility']:.2f}",
                    'Signal Strength': {
                        'Overall': f"{scenario['signal_strength']:.2f}",
                        'Bollinger': f"{scenario['technical_indicators']['bb_signal']}",
                        'RSI': f"{scenario['technical_indicators']['rsi_signal']}",
                        'MACD': f"{scenario['technical_indicators']['macd_signal']}"
                    }
                },
                'Support and Resistance': {
                    'Support': f"{scenario['support_and_resistance']['support']:.2f}",
                    'Resistance': f"{scenario['support_and_resistance']['resistance']:.2f}",
                    'Price in Range': f"{scenario['support_and_resistance']['price_in_range']:.2f}"
                },
                'Forecast Time': (datetime.now() + timedelta(hours=scenario['hour'])).strftime('%Y-%m-%d %H:%M:%S')
            }
            suggestions.append(suggestion)
        
        return suggestions

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

    
    def generate_timeseries_forecast(self, current_data: pd.DataFrame) -> List[Dict]:
        """Generate detailed time series forecasts with enhanced error handling"""
        try:
            intervals = 24  # 2 hours * 12 intervals per hour (5-min each)
            df_indicators = self.calculate_indicators(current_data)
            
            if self.lstm_model is None or self.rl_model is None:
                raise ValueError("Models not trained. Call train_models first.")
                
            # Prepare base data with safety checks
            recent_data = df_indicators[self.features].tail(self.lookback_period)
            recent_data = recent_data.replace([np.inf, -np.inf], np.nan).ffill().bfill()
            
            # Initialize with safe values
            current_price = max(float(current_data['close'].iloc[-1]), 0.01)  # Ensure non-zero price
            initial_volume = max(float(current_data['volume'].iloc[-1]), 1.0)  # Ensure non-zero volume
            
            # Initialize forecast arrays with safe values
            price_sequence = [current_price]
            volume_sequence = [initial_volume]
            
            # Get base statistics safely
            base_volatility = max(float(recent_data['volatility'].mean()), 0.001)  # Ensure non-zero volatility
            volume_mean = max(float(current_data['volume'].mean()), 1.0)
            volume_std = max(float(current_data['volume'].std()), 0.1)
            
            base_scaled_data = self.scaler.transform(recent_data)
            X = base_scaled_data.reshape(1, self.lookback_period, len(self.features))
            latest_indicators = df_indicators.iloc[-1].copy()
            
            # Initialize output containers
            forecasts = []
            
            for interval in range(intervals):
                try:
                    timestamp = datetime.now() + timedelta(minutes=5 * (interval + 1))
                    
                    # Generate base prediction
                    if interval == 0:
                        base_pred = self.lstm_model.predict(X, verbose=0)[0][0]
                        dummy_features = np.zeros((1, len(self.features)))
                        dummy_features[0, 0] = base_pred
                        pred_price = max(float(self.scaler.inverse_transform(dummy_features)[0][0]), 0.01)
                        scaled_prev_pred = base_pred
                    else:
                        X = np.roll(X, -1, axis=1)
                        X[0, -1] = scaled_prev_pred
                        base_pred = self.lstm_model.predict(X, verbose=0)[0][0]
                        dummy_features = np.zeros((1, len(self.features)))
                        dummy_features[0, 0] = base_pred
                        pred_price = max(float(self.scaler.inverse_transform(dummy_features)[0][0]), 0.01)
                    
                    # Add noise with safety checks
                    time_factor = np.sqrt(interval + 1)
                    price_noise = np.random.normal(0, base_volatility * 0.1 * time_factor)
                    pred_price *= (1 + np.clip(price_noise, -0.1, 0.1))  # Limit noise impact
                    
                    # Generate predicted volume safely
                    volume_noise = np.clip(np.random.normal(0, 0.1), -0.5, 0.5)
                    pred_volume = max(volume_mean * (1 + volume_noise), 1.0)
                    
                    # Store sequences
                    price_sequence.append(pred_price)
                    volume_sequence.append(pred_volume)
                    
                    # Calculate support and resistance levels safely
                    support_level = min(price_sequence) * 0.95  # More conservative levels
                    resistance_level = max(price_sequence) * 1.05
                    
                    # Update indicators
                    updated_indicators = self._update_indicators_sequence(
                        latest_indicators,
                        price_sequence,
                        volume_sequence,
                        interval
                    )
                    
                    # Get RL model action
                    state = self._get_state(updated_indicators)
                    action, _ = self.rl_model.predict(state, deterministic=True)
                    
                    forecast = {
                        'interval': interval + 1,
                        'timestamp': timestamp,
                        'price': {
                            'current': current_price,
                            'predicted': pred_price,
                            'support': support_level,
                            'resistance': resistance_level,
                            'volatility': base_volatility * time_factor
                        },
                        'volume': {
                            'predicted': pred_volume,
                            'trend': 'Increasing' if pred_volume > volume_sequence[-2] else 'Decreasing',
                            'strength': max(0, abs(pred_volume - volume_sequence[-2]) / max(volume_sequence[-2], 1))
                        },
                        'indicators': {
                            'rsi': float(updated_indicators['rsi']),
                            'macd': float(updated_indicators.get('macd', 0)),
                            'bb_position': float(updated_indicators.get('bb_position', 0.5)),
                            'atr': float(updated_indicators.get('atr', base_volatility))
                        },
                        'signals': {
                            'action': ['BUY', 'SELL', 'HOLD'][int(action)],
                            'strength': float(updated_indicators.get('trend_strength', 0.5)),
                            'confidence': float(updated_indicators.get('market_strength', 0.5))
                        }
                    }
                    
                    forecasts.append(forecast)
                    scaled_prev_pred = base_pred
                    
                except Exception as e:
                    self.logger.warning(f"Error in interval {interval}: {str(e)}")
                    continue
            
            return {
                'forecasts': forecasts,
                'sequences': {
                    'timestamps': [(datetime.now() + timedelta(minutes=5*i)).strftime('%H:%M') 
                                for i in range(len(forecasts)+1)],
                    'prices': price_sequence,
                    'volumes': volume_sequence
                }
            }
                
        except Exception as e:
            self.logger.error(f"Error in generate_timeseries_forecast: {str(e)}")
            raise

    def _update_indicators_sequence(self, indicators: pd.Series, 
                              price_sequence: List[float], 
                              volume_sequence: List[float], 
                              interval: int) -> pd.Series:
        """Update technical indicators based on predicted price and volume sequence with error handling"""
        try:
            updated = indicators.copy()
            
            # Calculate returns safely
            price_sequence = np.array(price_sequence)
            volume_sequence = np.array(volume_sequence)
            
            # Ensure we have valid sequences
            if len(price_sequence) < 2 or len(volume_sequence) < 2:
                return indicators
                
            # Calculate returns with safety checks
            price_diff = np.diff(price_sequence)
            price_prev = price_sequence[:-1]
            returns = np.divide(price_diff, price_prev, out=np.zeros_like(price_diff), where=price_prev!=0)
            
            # Update price-based indicators
            updated['close'] = float(price_sequence[-1])
            updated['volatility'] = float(np.std(returns)) if len(returns) > 1 else float(indicators['volatility'])
            updated['price_momentum'] = float(returns[-1]) if returns.size > 0 else 0.0
            
            # Update volume-based indicators safely
            if volume_sequence[-2] != 0:
                volume_change = (volume_sequence[-1] - volume_sequence[-2]) / volume_sequence[-2]
            else:
                volume_change = 0.0
            updated['volume_momentum'] = float(volume_change)
            
            # Update trend indicators
            if len(price_sequence) >= 20:
                updated['sma_20'] = float(np.mean(price_sequence[-20:]))
            if len(price_sequence) >= 50:
                updated['sma_50'] = float(np.mean(price_sequence[-50:]))
            
            # Update RSI (simplified) with safety checks
            gains = np.sum([x for x in returns if x > 0]) if returns.size > 0 else 0
            losses = abs(np.sum([x for x in returns if x < 0])) if returns.size > 0 else 0
            
            if losses != 0:
                rs = gains / losses
                updated['rsi'] = min(100.0, max(0.0, 100.0 - (100.0 / (1.0 + rs))))
            else:
                updated['rsi'] = 100.0 if gains > 0 else 50.0
            
            # Update trend strength safely
            if returns.size > 0:
                look_back = min(6, len(returns))
                trend_signals = [1 if returns[-i] > 0 else 0 for i in range(1, look_back+1)]
                updated['trend_strength'] = float(np.mean(trend_signals))
            else:
                updated['trend_strength'] = 0.5
                
            # Ensure all values are finite
            for key in updated.index:
                if not np.isfinite(updated[key]):
                    updated[key] = indicators[key]
            
            return updated
            
        except Exception as e:
            self.logger.warning(f"Error in _update_indicators_sequence: {str(e)}")
            return indicators


    def plot_forecasts(self, forecast_data: Dict):
        """Create visualization of the forecasted time series"""
        import matplotlib.pyplot as plt
        
        timestamps = forecast_data['sequences']['timestamps']
        prices = forecast_data['sequences']['prices']
        volumes = forecast_data['sequences']['volumes']
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # Plot price forecasts
        ax1.plot(timestamps, prices, 'b-', label='Predicted Price')
        ax1.set_title('Price Forecast - Next 2 Hours (5-min intervals)')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Price')
        ax1.grid(True)
        
        # Add support and resistance lines
        support_level = min(prices) * 0.995
        resistance_level = max(prices) * 1.005
        ax1.axhline(y=support_level, color='g', linestyle='--', label='Support')
        ax1.axhline(y=resistance_level, color='r', linestyle='--', label='Resistance')
        
        # Plot volume forecasts
        ax2.bar(timestamps, volumes, alpha=0.6, label='Predicted Volume')
        ax2.set_title('Volume Forecast - Next 2 Hours (5-min intervals)')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Volume')
        ax2.grid(True)
        
        # Rotate x-axis labels for better readability
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        # Add legends
        ax1.legend()
        ax2.legend()
        
        plt.tight_layout()
        return plt.gcf()

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
