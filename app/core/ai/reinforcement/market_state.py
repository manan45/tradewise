from typing import List, Dict, Optional
import numpy as np
import pandas as pd
import logging

class MarketState:
    """Manages market state representation for RL"""
    
    def __init__(self):
        self.state_features = [
            'trend_strength',
            'trend_consistency', 
            'rsi',
            'macd',
            'support_distance',
            'resistance_distance',
            'volatility',
            'psychological_sentiment',
            'zone_strength',
            'volume_pressure'
        ]
        
    def get_state_features(self, data: pd.DataFrame) -> np.ndarray:
        """Get current market state features"""
        try:
            features = pd.DataFrame()
            
            # Technical features
            features['trend_strength'] = self._calculate_trend_strength(data)
            features['trend_consistency'] = self._calculate_trend_consistency(data)
            features['rsi'] = data.get('rsi', pd.Series([50] * len(data)))
            features['macd'] = data.get('macd', pd.Series([0] * len(data)))
            
            # Support/Resistance features
            sr_levels = self._calculate_sr_levels(data)
            features['support_distance'] = self._calculate_level_distance(data, sr_levels['support'])
            features['resistance_distance'] = self._calculate_level_distance(data, sr_levels['resistance'])
            
            # Additional features
            features['volatility'] = self._calculate_volatility(data)
            features['psychological_sentiment'] = self._calculate_sentiment(data)
            features['zone_strength'] = self._calculate_zone_strength(data)
            features['volume_pressure'] = self._calculate_volume_pressure(data)
            
            # Return latest state
            return features.iloc[-1].values
            
        except Exception as e:
            logging.error(f"Error getting state features: {str(e)}")
            return np.zeros(len(self.state_features))

    def _calculate_trend_strength(self, data: pd.DataFrame) -> pd.Series:
        """Calculate trend strength using moving averages"""
        try:
            ma20 = data['close'].rolling(window=20).mean()
            ma50 = data['close'].rolling(window=50).mean()
            
            trend_strength = (ma20 - ma50) / ma50
            return trend_strength.clip(-1, 1)
            
        except Exception as e:
            logging.error(f"Error calculating trend strength: {str(e)}")
            return pd.Series([0] * len(data))

    def _calculate_trend_consistency(self, data: pd.DataFrame) -> pd.Series:
        """Calculate trend consistency"""
        try:
            short_ma = data['close'].rolling(window=20).mean()
            long_ma = data['close'].rolling(window=50).mean()            
            trend_consistency = (short_ma - long_ma) / long_ma
            
            return trend_consistency.clip(-1, 1)
            
        except Exception as e:
            logging.error(f"Error calculating trend consistency: {str(e)}")
            return pd.Series([0] * len(data))

    def _calculate_sr_levels(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate support and resistance levels"""
        try:
            window = 20
            highs = data['high'].rolling(window=window).max()
            lows = data['low'].rolling(window=window).min()
            
            current_price = data['close'].iloc[-1]
            
            # Find nearest support and resistance
            support = lows[lows < current_price].iloc[-1] if len(lows[lows < current_price]) > 0 else current_price * 0.95
            resistance = highs[highs > current_price].iloc[-1] if len(highs[highs > current_price]) > 0 else current_price * 1.05
            
            return {'support': support, 'resistance': resistance}
            
        except Exception as e:
            logging.error(f"Error calculating S/R levels: {str(e)}")
            return {'support': 0.0, 'resistance': 0.0}

    def _calculate_level_distance(self, data: pd.DataFrame, level: float) -> pd.Series:
        """Calculate distance to support/resistance level"""
        try:
            return ((data['close'] - level) / level).clip(-1, 1)
        except Exception as e:
            logging.error(f"Error calculating level distance: {str(e)}")
            return pd.Series([0] * len(data))

    def _calculate_volatility(self, data: pd.DataFrame) -> pd.Series:
        """Calculate price volatility"""
        try:
            returns = data['close'].pct_change()
            volatility = returns.rolling(window=20).std()
            return volatility.clip(0, 1)
        except Exception as e:
            logging.error(f"Error calculating volatility: {str(e)}")
            return pd.Series([0] * len(data))

    def _calculate_sentiment(self, data: pd.DataFrame) -> pd.Series:
        """Calculate market sentiment"""
        try:
            # Simple sentiment based on price momentum and volume
            returns = data['close'].pct_change()
            volume_change = data['volume'].pct_change()
            
            sentiment = (
                returns.rolling(window=10).mean() * 0.7 +
                volume_change.rolling(window=10).mean() * 0.3
            )
            
            return sentiment.clip(-1, 1)
        except Exception as e:
            logging.error(f"Error calculating sentiment: {str(e)}")
            return pd.Series([0] * len(data))

    def _calculate_zone_strength(self, data: pd.DataFrame) -> pd.Series:
        """Calculate price zone strength"""
        try:
            # Zone strength based on price consolidation
            typical_price = (data['high'] + data['low'] + data['close']) / 3
            price_range = (data['high'] - data['low']) / data['close']
            
            zone_strength = 1 - price_range.rolling(window=10).mean()
            return zone_strength.clip(0, 1)
        except Exception as e:
            logging.error(f"Error calculating zone strength: {str(e)}")
            return pd.Series([0] * len(data))

    def _calculate_volume_pressure(self, data: pd.DataFrame) -> pd.Series:
        """Calculate volume pressure"""
        try:
            # Volume pressure based on volume and price direction
            volume_ma = data['volume'].rolling(window=20).mean()
            price_direction = np.sign(data['close'].pct_change())
            
            volume_pressure = (data['volume'] / volume_ma - 1) * price_direction
            return volume_pressure.clip(-1, 1)
        except Exception as e:
            logging.error(f"Error calculating volume pressure: {str(e)}")
            return pd.Series([0] * len(data))
