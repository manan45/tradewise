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

