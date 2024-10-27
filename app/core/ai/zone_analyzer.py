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
            },
            'fibonacci': {
                'strong': 0.7,
                'weak': 0.3,
                'levels': [0.236, 0.382, 0.5, 0.618, 0.786]
            }
        }

    def analyze(self, state_history: List[Dict], trades: List[Dict]) -> Dict:
        """Analyze zone patterns from market data"""
        try:
            patterns = {
                'support_zones': self._analyze_support_zones(state_history),
                'resistance_zones': self._analyze_resistance_zones(state_history),
                'consolidation_zones': self._analyze_consolidation_zones(state_history),
                'breakout_zones': self._analyze_breakout_zones(state_history, trades),
                'fibonacci_zones': self._analyze_fibonacci_zones(state_history)
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

    def _identify_support_levels(self, prices: List[float], lows: List[float]) -> List[float]:
        """Identify support levels using local minima"""
        support_levels = []
        window_size = 10
        
        for i in range(window_size, len(lows) - window_size):
            window = lows[i-window_size:i+window_size]
            if min(window) == lows[i]:
                support_levels.append(lows[i])
                
        return sorted(list(set(support_levels)))

    def _calculate_support_strength(self, prices: List[float], support_levels: List[float]) -> float:
        """Calculate strength of support levels"""
        if not support_levels:
            return 0.0
            
        bounces = 0
        tests = 0
        
        for price_idx in range(1, len(prices)):
            for level in support_levels:
                if abs(prices[price_idx-1] - level) / level < 0.01:  # Within 1% of support
                    tests += 1
                    if prices[price_idx] > level:
                        bounces += 1
                        
        return bounces / max(tests, 1)

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

    def _identify_resistance_levels(self, prices: List[float], highs: List[float]) -> List[float]:
        """Identify resistance levels using local maxima"""
        resistance_levels = []
        window_size = 10
        
        for i in range(window_size, len(highs) - window_size):
            window = highs[i-window_size:i+window_size]
            if max(window) == highs[i]:
                resistance_levels.append(highs[i])
                
        return sorted(list(set(resistance_levels)))

    def _calculate_resistance_strength(self, prices: List[float], resistance_levels: List[float]) -> float:
        """Calculate strength of resistance levels"""
        if not resistance_levels:
            return 0.0
            
        rejections = 0
        tests = 0
        
        for price_idx in range(1, len(prices)):
            for level in resistance_levels:
                if abs(prices[price_idx-1] - level) / level < 0.01:  # Within 1% of resistance
                    tests += 1
                    if prices[price_idx] < level:
                        rejections += 1
                        
        return rejections / max(tests, 1)

    def _analyze_fibonacci_zones(self, history: List[Dict]) -> Dict:
        """Analyze Fibonacci retracement zones"""
        try:
            prices = [state['technical_state']['close'] for state in history]
            highs = [state['technical_state']['high'] for state in history]
            lows = [state['technical_state']['low'] for state in history]
            
            fib_levels = self._calculate_fibonacci_levels(highs, lows)
            
            return {
                'levels': fib_levels,
                'strength': self._calculate_fib_zone_strength(prices, fib_levels),
                'tests': self._count_zone_tests(prices, fib_levels, 'fibonacci')
            }
        except Exception as e:
            logging.error(f"Error analyzing fibonacci zones: {str(e)}")
            return {}

    def _calculate_fibonacci_levels(self, highs: List[float], lows: List[float]) -> Dict[str, float]:
        """Calculate Fibonacci retracement levels"""
        high = max(highs[-50:])  # Use recent high
        low = min(lows[-50:])    # Use recent low
        diff = high - low
        
        levels = {}
        for fib in self.zone_thresholds['fibonacci']['levels']:
            levels[str(fib)] = high - (diff * fib)
            
        return levels

    def _calculate_fib_zone_strength(self, prices: List[float], fib_levels: Dict[str, float]) -> float:
        """Calculate strength of Fibonacci zones"""
        if not fib_levels:
            return 0.0
            
        reactions = 0
        tests = 0
        
        for price_idx in range(1, len(prices)):
            for level in fib_levels.values():
                if abs(prices[price_idx-1] - level) / level < 0.01:  # Within 1% of fib level
                    tests += 1
                    if abs(prices[price_idx] - prices[price_idx-1]) / prices[price_idx-1] > 0.002:  # 0.2% reaction
                        reactions += 1
                        
        return reactions / max(tests, 1)

    def _count_zone_tests(self, prices: List[float], levels: List[float], zone_type: str) -> int:
        """Count number of times a zone has been tested"""
        tests = 0
        for price in prices:
            for level in levels:
                if abs(price - level) / level < 0.01:  # Within 1% of level
                    tests += 1
        return tests

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
        
        # Support + Fibonacci combo signals
        support_strength = patterns['support_zones']['strength']
        fib_strength = patterns['fibonacci_zones']['strength']
        combo_strength = (support_strength + fib_strength) / 2
        
        if combo_strength > self.zone_thresholds['support']['strong']:
            signals.append({
                'type': 'support_fib_combo',
                'strength': combo_strength,
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
            'support_fib_combo': 0.7,  # 70% weight for support + fib combo
            'resistance': 0.15,
            'consolidation': 0.1,
            'breakout': 0.05
        }
        
        support_fib_strength = (
            patterns['support_zones'].get('strength', 0) + 
            patterns['fibonacci_zones'].get('strength', 0)
        ) / 2
        
        strength = (
            support_fib_strength * weights['support_fib_combo'] +
            patterns['resistance_zones'].get('strength', 0) * weights['resistance'] +
            patterns['consolidation_zones'].get('breakout_probability', 0) * weights['consolidation'] +
            (len(patterns['breakout_zones'].get('potential_breakouts', [])) > 0) * weights['breakout']
        )
        
        return float(np.clip(strength, 0, 1))

    def identify_zone(self, prices: List[float], window_size: int = 20) -> List[Dict]:
        """Identify price zones using statistical analysis"""
        zones = []
        
        # Calculate price statistics in rolling windows
        for i in range(window_size, len(prices)):
            window = prices[i-window_size:i]
            mean = np.mean(window)
            std = np.std(window)
            
            # Define zone boundaries
            upper = mean + std
            lower = mean - std
            
            # Calculate zone strength based on price adherence
            in_zone = sum(1 for p in window if lower <= p <= upper)
            strength = in_zone / window_size
            
            # Identify zone type
            if prices[i] > upper:
                zone_type = 'resistance'
            elif prices[i] < lower:
                zone_type = 'support'
            else:
                zone_type = 'consolidation'
                
            zones.append({
                'type': zone_type,
                'upper': upper,
                'lower': lower,
                'strength': strength,
                'mean': mean,
                'std': std
            })
            
        return zones
