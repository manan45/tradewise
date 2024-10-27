from typing import Dict, List
import numpy as np
import pandas as pd
import logging

class TechnicalPatternAnalyzer:
    """Analyzes technical patterns in market data"""
    
    def __init__(self):
        self.pattern_thresholds = {
            'trend': {
                'strong': 0.7,
                'weak': 0.3
            },
            'momentum': {
                'overbought': 0.8,
                'oversold': 0.2
            },
            'volatility': {
                'high': 0.7,
                'low': 0.3
            }
        }
        self.logger = logging.getLogger(__name__)

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for the dataset"""
        try:
            data = df.copy()
            
            # Moving Averages
            data['sma_20'] = data['close'].rolling(window=20).mean()
            data['sma_50'] = data['close'].rolling(window=50).mean()
            
            # RSI
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            data['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            exp1 = data['close'].ewm(span=12, adjust=False).mean()
            exp2 = data['close'].ewm(span=26, adjust=False).mean()
            data['macd'] = exp1 - exp2
            data['macd_hist'] = data['macd'] - data['macd'].ewm(span=9, adjust=False).mean()
            
            # Bollinger Bands
            data['bb_middle'] = data['close'].rolling(window=20).mean()
            bb_std = data['close'].rolling(window=20).std()
            data['bb_upper'] = data['bb_middle'] + (bb_std * 2)
            data['bb_lower'] = data['bb_middle'] - (bb_std * 2)
            data['bb_width'] = (data['bb_upper'] - data['bb_lower']) / data['bb_middle']
            data['bb_position'] = (data['close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'])
            
            # Volatility and Trend
            data['volatility'] = data['close'].rolling(window=20).std() / data['close'].rolling(window=20).mean()
            data['trend_strength'] = abs(data['sma_20'] - data['sma_50']) / data['sma_50']
            data['price_momentum'] = data['close'].pct_change(periods=10)
            
            # ATR
            high_low = data['high'] - data['low']
            high_close = abs(data['high'] - data['close'].shift())
            low_close = abs(data['low'] - data['close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            data['atr'] = true_range.rolling(window=14).mean()
            
            # Forward fill NaN values
            data = data.fillna(method='ffill').fillna(method='bfill')
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error calculating indicators: {str(e)}")
            return df

    def get_trend_strength(self, df: pd.DataFrame) -> float:
        """Get current trend strength"""
        try:
            if 'trend_strength' not in df.columns:
                df = self.calculate_indicators(df)
            return float(df['trend_strength'].iloc[-1])
        except Exception as e:
            self.logger.error(f"Error getting trend strength: {str(e)}")
            return 0.5

    def analyze_volume_consistency(self, df: pd.DataFrame) -> float:
        """Analyze volume consistency"""
        try:
            volume_ma = df['volume'].rolling(window=20).mean()
            volume_std = df['volume'].rolling(window=20).std()
            consistency = 1 - (volume_std / volume_ma).iloc[-1]
            return float(np.clip(consistency, 0, 1))
        except Exception as e:
            self.logger.error(f"Error analyzing volume consistency: {str(e)}")
            return 0.5

    def analyze(self, state_history: List[Dict], trades: List[Dict]) -> Dict:
        """Analyze technical patterns from market data"""
        try:
            patterns = {
                'trend_patterns': self._analyze_trend_patterns(state_history),
                'momentum_patterns': self._analyze_momentum_patterns(state_history),
                'volatility_patterns': self._analyze_volatility_patterns(state_history),
                'volume_patterns': self._analyze_volume_patterns(state_history)
            }
            
            return {
                'patterns': patterns,
                'signals': self._generate_signals(patterns),
                'strength': self._calculate_pattern_strength(patterns)
            }
            
        except Exception as e:
            logging.error(f"Error in technical analysis: {str(e)}")
            return {}

    def _analyze_trend_patterns(self, history: List[Dict]) -> Dict:
        """Analyze trend patterns"""
        try:
            prices = [state['technical_state']['close'] for state in history]
            
            return {
                'direction': 'up' if prices[-1] > prices[0] else 'down',
                'strength': self._calculate_trend_strength(prices),
                'consistency': self._calculate_trend_consistency(prices)
            }
        except Exception as e:
            logging.error(f"Error analyzing trends: {str(e)}")
            return {}

    def _analyze_momentum_patterns(self, history: List[Dict]) -> Dict:
        """Analyze momentum patterns"""
        try:
            momentum_values = [state['technical_state'].get('momentum', 0) for state in history]
            
            return {
                'current': momentum_values[-1],
                'trend': self._calculate_momentum_trend(momentum_values),
                'extremes': self._detect_momentum_extremes(momentum_values)
            }
        except Exception as e:
            logging.error(f"Error analyzing momentum: {str(e)}")
            return {}

    def _analyze_volatility_patterns(self, history: List[Dict]) -> Dict:
        """Analyze volatility patterns"""
        try:
            volatility = [state['technical_state'].get('volatility', 0) for state in history]
            
            return {
                'current_level': volatility[-1],
                'trend': self._calculate_volatility_trend(volatility),
                'regime': self._identify_volatility_regime(volatility)
            }
        except Exception as e:
            logging.error(f"Error analyzing volatility: {str(e)}")
            return {}

    def _analyze_volume_patterns(self, history: List[Dict]) -> Dict:
        """Analyze volume patterns"""
        try:
            volume = [state['technical_state'].get('volume', 0) for state in history]
            
            return {
                'current_level': volume[-1],
                'trend': self._calculate_volume_trend(volume),
                'spikes': self._detect_volume_spikes(volume)
            }
        except Exception as e:
            logging.error(f"Error analyzing volume: {str(e)}")
            return {}

    def _generate_signals(self, patterns: Dict) -> List[Dict]:
        """Generate trading signals from patterns"""
        signals = []
        
        # Trend signals
        if patterns['trend_patterns']['strength'] > self.pattern_thresholds['trend']['strong']:
            signals.append({
                'type': 'trend',
                'direction': patterns['trend_patterns']['direction'],
                'strength': patterns['trend_patterns']['strength'],
                'confidence': 'high'
            })
            
        # Momentum signals
        momentum = patterns['momentum_patterns']
        if momentum['current'] > self.pattern_thresholds['momentum']['overbought']:
            signals.append({
                'type': 'momentum',
                'signal': 'overbought',
                'strength': momentum['current'],
                'confidence': 'medium'
            })
        elif momentum['current'] < self.pattern_thresholds['momentum']['oversold']:
            signals.append({
                'type': 'momentum',
                'signal': 'oversold',
                'strength': momentum['current'],
                'confidence': 'medium'
            })
            
        return signals

    def _calculate_pattern_strength(self, patterns: Dict) -> float:
        """Calculate overall pattern strength"""
        weights = {
            'trend': 0.4,
            'momentum': 0.3,
            'volatility': 0.2,
            'volume': 0.1
        }
        
        strength = (
            patterns['trend_patterns'].get('strength', 0) * weights['trend'] +
            patterns['momentum_patterns'].get('current', 0) * weights['momentum'] +
            patterns['volatility_patterns'].get('current_level', 0) * weights['volatility'] +
            patterns['volume_patterns'].get('current_level', 0) * weights['volume']
        )
        
        return float(np.clip(strength, 0, 1))
