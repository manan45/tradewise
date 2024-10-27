from typing import Dict, List
import numpy as np
import pandas as pd
import logging

class ZonePatternAnalyzer:
    """Analyzes price zone patterns"""
    
    def __init__(self):
        self.zone_thresholds = {
            'support': {
                'strong': 0.7,
                'weak': 0.3
            },
            'resistance': {
                'strong': 0.7,
                'weak': 0.3
            },
            'consolidation': {
                'min_periods': 5,
                'max_range': 0.02
            }
        }

    def analyze(self, state_history: List[Dict], trades: List[Dict]) -> Dict:
        """Analyze zone patterns from market data"""
        try:
            patterns = {
                'support_zones': self._analyze_support_zones(state_history),
                'resistance_zones': self._analyze_resistance_zones(state_history),
                'consolidation_zones': self._analyze_consolidation_zones(state_history),
                'breakout_zones': self._analyze_breakout_zones(state_history, trades)
            }
            
            return {
                'patterns': patterns,
                'signals': self._generate_zone_signals(patterns),
                'strength': self._calculate_zone_strength(patterns)
            }
            
        except Exception as e:
            logging.error(f"Error in zone analysis: {str(e)}")
            return {}

    def _analyze_support_zones(self, history: List[Dict]) -> Dict:
        """Analyze support zones"""
        try:
            prices = [state['technical_state']['close'] for state in history]
            lows = [state['technical_state']['low'] for state in history]
            
            support_levels = self._identify_support_levels(prices, lows)
            
            return {
                'levels': support_levels,
                'strength': self._calculate_support_strength(prices, support_levels),
                'tests': self._count_zone_tests(prices, support_levels, 'support')
            }
        except Exception as e:
            logging.error(f"Error analyzing support zones: {str(e)}")
            return {}

    def _analyze_resistance_zones(self, history: List[Dict]) -> Dict:
        """Analyze resistance zones"""
        try:
            prices = [state['technical_state']['close'] for state in history]
            highs = [state['technical_state']['high'] for state in history]
            
            resistance_levels = self._identify_resistance_levels(prices, highs)
            
            return {
                'levels': resistance_levels,
                'strength': self._calculate_resistance_strength(prices, resistance_levels),
                'tests': self._count_zone_tests(prices, resistance_levels, 'resistance')
            }
        except Exception as e:
            logging.error(f"Error analyzing resistance zones: {str(e)}")
            return {}

    def _analyze_consolidation_zones(self, history: List[Dict]) -> Dict:
        """Analyze consolidation zones"""
        try:
            prices = [state['technical_state']['close'] for state in history]
            
            return {
                'zones': self._identify_consolidation_zones(prices),
                'current_zone': self._get_current_consolidation(prices),
                'breakout_probability': self._calculate_breakout_probability(prices)
            }
        except Exception as e:
            logging.error(f"Error analyzing consolidation zones: {str(e)}")
            return {}

    def _analyze_breakout_zones(self, history: List[Dict], trades: List[Dict]) -> Dict:
        """Analyze breakout zones"""
        try:
            prices = [state['technical_state']['close'] for state in history]
            volumes = [state['technical_state'].get('volume', 0) for state in history]
            
            return {
                'recent_breakouts': self._identify_breakouts(prices, volumes),
                'failed_breakouts': self._identify_failed_breakouts(prices, trades),
                'potential_breakouts': self._identify_potential_breakouts(prices, volumes)
            }
        except Exception as e:
            logging.error(f"Error analyzing breakout zones: {str(e)}")
            return {}

    def _generate_zone_signals(self, patterns: Dict) -> List[Dict]:
        """Generate trading signals from zone patterns"""
        signals = []
        
        # Support signals
        if patterns['support_zones']['strength'] > self.zone_thresholds['support']['strong']:
            signals.append({
                'type': 'support',
                'strength': patterns['support_zones']['strength'],
                'confidence': 'high',
                'action': 'buy'
            })
            
        # Resistance signals
        if patterns['resistance_zones']['strength'] > self.zone_thresholds['resistance']['strong']:
            signals.append({
                'type': 'resistance',
                'strength': patterns['resistance_zones']['strength'],
                'confidence': 'high',
                'action': 'sell'
            })
            
        # Breakout signals
        breakouts = patterns['breakout_zones']['potential_breakouts']
        if breakouts:
            signals.append({
                'type': 'breakout',
                'direction': breakouts[-1]['direction'],
                'strength': breakouts[-1]['strength'],
                'confidence': 'medium'
            })
            
        return signals

    def _calculate_zone_strength(self, patterns: Dict) -> float:
        """Calculate overall zone pattern strength"""
        weights = {
            'support': 0.3,
            'resistance': 0.3,
            'consolidation': 0.2,
            'breakout': 0.2
        }
        
        strength = (
            patterns['support_zones'].get('strength', 0) * weights['support'] +
            patterns['resistance_zones'].get('strength', 0) * weights['resistance'] +
            patterns['consolidation_zones'].get('breakout_probability', 0) * weights['consolidation'] +
            (len(patterns['breakout_zones'].get('potential_breakouts', [])) > 0) * weights['breakout']
        )
        
        return float(np.clip(strength, 0, 1))
