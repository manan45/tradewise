import pandas as pd
import numpy as np
from ta.trend import MACD, EMAIndicator, SMAIndicator, ADXIndicator
from ta.momentum import RSIIndicator, StochRSIIndicator, WilliamsRIndicator
from ta.volatility import BollingerBands, AverageTrueRange, KeltnerChannel
from ta.volume import AccDistIndexIndicator, OnBalanceVolumeIndicator, ForceIndexIndicator
from ta.others import DailyReturnIndicator, CumulativeReturnIndicator
from typing import List, Dict
import warnings

class PsychologyPatternAnalyzer:
    """Analyzes psychological patterns in trading sessions"""
    
    def __init__(self):
        self.emotional_thresholds = {
            'extreme_fear': 0.2,
            'extreme_greed': 0.8,
            'balanced': (0.4, 0.6)
        }
        self.confidence_thresholds = {
            'low': 0.3,
            'high': 0.7
        }
        
    def analyze(self, state_history: List[Dict], trades: List[Dict]) -> Dict:
        """Analyze psychological patterns from session history"""
        patterns = {
            'emotional_patterns': self._analyze_emotional_patterns(state_history),
            'confidence_patterns': self._analyze_confidence_patterns(state_history),
            'decision_patterns': self._analyze_decision_patterns(state_history, trades),
            'bias_patterns': self._analyze_bias_patterns(state_history, trades)
        }
        
        return {
            'patterns': patterns,
            'recommendations': self._generate_recommendations(patterns)
        }
    
    def _analyze_emotional_patterns(self, history: List[Dict]) -> Dict:
        """Analyze emotional state patterns"""
        emotional_states = [state['psychological_state'].get('emotional_balance', 0.5) 
                          for state in history]
        
        # Detect emotional cycles
        cycles = self._detect_cycles(emotional_states)
        
        # Analyze emotional stability
        stability = np.std(emotional_states)
        
        # Detect emotional extremes
        extremes = {
            'fear_episodes': sum(1 for e in emotional_states if e < self.emotional_thresholds['extreme_fear']),
            'greed_episodes': sum(1 for e in emotional_states if e > self.emotional_thresholds['extreme_greed']),
            'balanced_periods': sum(1 for e in emotional_states 
                                  if self.emotional_thresholds['balanced'][0] <= e <= self.emotional_thresholds['balanced'][1])
        }
        
        return {
            'cycles': cycles,
            'stability': stability,
            'extremes': extremes
        }
    
    def _analyze_confidence_patterns(self, history: List[Dict]) -> Dict:
        """Analyze confidence patterns"""
        confidence_levels = [state['psychological_state'].get('confidence', 0.5) 
                           for state in history]
        
        # Analyze confidence trends
        confidence_trend = self._calculate_trend(confidence_levels)
        
        # Detect confidence shifts
        shifts = self._detect_significant_shifts(confidence_levels)
        
        return {
            'trend': confidence_trend,
            'shifts': shifts,
            'average': np.mean(confidence_levels),
            'stability': np.std(confidence_levels)
        }
    
    def _analyze_decision_patterns(self, history: List[Dict], trades: List[Dict]) -> Dict:
        """Analyze decision-making patterns"""
        decisions = []
        for state, trade in zip(history, trades):
            if trade:
                decisions.append({
                    'emotional_state': state['psychological_state'].get('emotional_balance', 0.5),
                    'confidence': state['psychological_state'].get('confidence', 0.5),
                    'outcome': trade['pnl']
                })
        
        return {
            'emotion_correlation': self._calculate_correlation([d['emotional_state'] for d in decisions],
                                                            [d['outcome'] for d in decisions]),
            'confidence_correlation': self._calculate_correlation([d['confidence'] for d in decisions],
                                                               [d['outcome'] for d in decisions])
        }
    
    def _detect_cycles(self, values: List[float]) -> Dict:
        """Detect cycles in time series data"""
        # Use FFT to detect cycles
        if len(values) < 2:
            return {'cycles': [], 'dominant_cycle': None}
            
        fft = np.fft.fft(values)
        freqs = np.fft.fftfreq(len(values))
        
        # Find dominant cycles
        dominant_cycles = []
        for freq, amp in zip(freqs, np.abs(fft)):
            if freq > 0:  # Only positive frequencies
                period = 1/freq if freq != 0 else 0
                dominant_cycles.append((period, amp))
        
        # Sort by amplitude
        dominant_cycles.sort(key=lambda x: x[1], reverse=True)
        
        return {
            'cycles': dominant_cycles[:3],  # Top 3 cycles
            'dominant_cycle': dominant_cycles[0] if dominant_cycles else None
        }

class TechnicalPatternAnalyzer:
    """Analyzes technical patterns in trading sessions"""
    
    def __init__(self):
        self.pattern_definitions = {
            'trend_reversal': {
                'window': 5,
                'threshold': 0.02
            },
            'breakout': {
                'volume_threshold': 1.5,
                'price_threshold': 0.01
            },
            'consolidation': {
                'range_threshold': 0.005,
                'min_periods': 3
            }
        }
    
    def analyze(self, state_history: List[Dict], trades: List[Dict]) -> Dict:
        """Analyze technical patterns from session history"""
        patterns = {
            'trend_patterns': self._analyze_trend_patterns(state_history),
            'volume_patterns': self._analyze_volume_patterns(state_history),
            'momentum_patterns': self._analyze_momentum_patterns(state_history),
            'volatility_patterns': self._analyze_volatility_patterns(state_history)
        }
        
        return {
            'patterns': patterns,
            'signals': self._generate_signals(patterns),
            'recommendations': self._generate_recommendations(patterns)
        }
    
    def _analyze_trend_patterns(self, history: List[Dict]) -> Dict:
        """Analyze trend patterns"""
        prices = [state['technical_state']['trend']['close'] for state in history]
        trend_strengths = [state['technical_state']['trend']['strength'] for state in history]
        
        # Detect trend changes
        trend_changes = self._detect_trend_changes(prices, self.pattern_definitions['trend_reversal'])
        
        # Analyze trend strength
        trend_strength_analysis = {
            'average': np.mean(trend_strengths),
            'consistency': np.std(trend_strengths),
            'direction': 'up' if prices[-1] > prices[0] else 'down'
        }
        
        return {
            'changes': trend_changes,
            'strength': trend_strength_analysis,
            'persistent_trend': self._is_persistent_trend(trend_strengths)
        }
    
    def _analyze_volume_patterns(self, history: List[Dict]) -> Dict:
        """Analyze volume patterns"""
        volumes = [state['technical_state'].get('volume', 0) for state in history]
        
        # Detect volume spikes
        spikes = self._detect_volume_spikes(volumes, self.pattern_definitions['breakout']['volume_threshold'])
        
        # Volume trend
        volume_trend = self._calculate_trend(volumes)
        
        return {
            'spikes': spikes,
            'trend': volume_trend,
            'average': np.mean(volumes),
            'distribution': self._analyze_volume_distribution(volumes)
        }
    
    def _detect_trend_changes(self, prices: List[float], params: Dict) -> List[Dict]:
        """Detect significant trend changes"""
        changes = []
        for i in range(params['window'], len(prices)):
            window_prices = prices[i-params['window']:i]
            current_price = prices[i]
            
            # Calculate price change
            price_change = (current_price - window_prices[0]) / window_prices[0]
            
            if abs(price_change) > params['threshold']:
                changes.append({
                    'index': i,
                    'magnitude': price_change,
                    'direction': 'up' if price_change > 0 else 'down'
                })
                
        return changes

class ZonePatternAnalyzer:
    """Analyzes price zone patterns in trading sessions"""
    
    def __init__(self):
        self.zone_params = {
            'support_resistance': {
                'touch_threshold': 3,
                'bounce_threshold': 0.001,
                'break_threshold': 0.002
            },
            'consolidation': {
                'range_threshold': 0.005,
                'min_periods': 5
            }
        }
    
    def analyze(self, state_history: List[Dict], trades: List[Dict]) -> Dict:
        """Analyze zone patterns from session history"""
        patterns = {
            'zone_interactions': self._analyze_zone_interactions(state_history),
            'zone_strength': self._analyze_zone_strength(state_history),
            'zone_transitions': self._analyze_zone_transitions(state_history),
            'trading_zones': self._analyze_trading_zones(state_history, trades)
        }
        
        return {
            'patterns': patterns,
            'key_zones': self._identify_key_zones(patterns),
            'recommendations': self._generate_zone_recommendations(patterns)
        }
    
    def _analyze_zone_interactions(self, history: List[Dict]) -> Dict:
        """Analyze how price interacts with zones"""
        interactions = []
        
        for i in range(1, len(history)):
            prev_state = history[i-1]
            curr_state = history[i]
            
            # Detect zone touches, bounces, and breaks
            for zone_type in ['support', 'resistance']:
                interaction = self._detect_zone_interaction(
                    prev_state['zone_state'],
                    curr_state['zone_state'],
                    zone_type
                )
                if interaction:
                    interactions.append(interaction)
        
        return self._summarize_interactions(interactions)
    
    def _detect_zone_interaction(self, prev_state: Dict, curr_state: Dict, zone_type: str) -> Dict:
        """Detect specific zone interactions"""
        price = curr_state['current_price']
        zone_level = curr_state[f'nearest_{zone_type}']
        
        # Calculate price distance from zone
        distance = abs(price - zone_level) / zone_level
        
        if distance < self.zone_params['support_resistance']['bounce_threshold']:
            return {
                'type': 'touch',
                'zone_type': zone_type,
                'price': price,
                'zone_level': zone_level,
                'strength': 1 - distance/self.zone_params['support_resistance']['bounce_threshold']
            }
        
        # Detect breaks
        prev_price = prev_state['current_price']
        if (zone_type == 'support' and price < zone_level and prev_price > zone_level) or \
           (zone_type == 'resistance' and price > zone_level and prev_price < zone_level):
            return {
                'type': 'break',
                'zone_type': zone_type,
                'price': price,
                'zone_level': zone_level,
                'magnitude': abs(price - zone_level) / zone_level
            }
        
        return None
    
    def _analyze_zone_strength(self, history: List[Dict]) -> Dict:
        """Analyze the strength of different price zones"""
        support_tests = []
        resistance_tests = []
        
        for state in history:
            if state['zone_state'].get('in_support_zone'):
                support_tests.append({
                    'level': state['zone_state']['nearest_support'],
                    'strength': state['zone_state']['zone_strength']
                })
            if state['zone_state'].get('in_resistance_zone'):
                resistance_tests.append({
                    'level': state['zone_state']['nearest_resistance'],
                    'strength': state['zone_state']['zone_strength']
                })
        
        return {
            'support': self._calculate_zone_strength(support_tests),
            'resistance': self._calculate_zone_strength(resistance_tests)
        }
    
    def _calculate_zone_strength(self, tests: List[Dict]) -> Dict:
        """Calculate zone strength metrics"""
        if not tests:
            return {'strength': 0, 'reliability': 0}
            
        strengths = [test['strength'] for test in tests]
        
        return {
            'strength': np.mean(strengths),
            'reliability': len(tests) / max(1, np.std(strengths)),
            'tests': len(tests)
        }
