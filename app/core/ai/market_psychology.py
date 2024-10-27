from typing import List, Dict
import numpy as np

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
        """Analyze confidence level patterns"""
        confidence_levels = [state['psychological_state'].get('confidence', 0.5)
                           for state in history]

        # Detect confidence trends
        trends = self._detect_trends(confidence_levels)
        
        # Analyze confidence stability
        stability = np.std(confidence_levels)
        
        # Detect confidence extremes
        extremes = {
            'low_confidence': sum(1 for c in confidence_levels if c < self.confidence_thresholds['low']),
            'high_confidence': sum(1 for c in confidence_levels if c > self.confidence_thresholds['high'])
        }
        
        return {
            'trends': trends,
            'stability': stability,
            'extremes': extremes
        }

    def _analyze_decision_patterns(self, history: List[Dict], trades: List[Dict]) -> Dict:
        """Analyze trading decision patterns"""
        # Extract decision metrics
        decision_times = [trade.get('decision_time', 0) for trade in trades]
        risk_levels = [trade.get('risk_level', 0.5) for trade in trades]
        
        # Analyze decision speed
        avg_decision_time = np.mean(decision_times) if decision_times else 0
        decision_consistency = np.std(decision_times) if decision_times else 0
        
        # Analyze risk patterns
        risk_patterns = {
            'risk_seeking': sum(1 for r in risk_levels if r > 0.7),
            'risk_averse': sum(1 for r in risk_levels if r < 0.3),
            'balanced_risk': sum(1 for r in risk_levels if 0.3 <= r <= 0.7)
        }
        
        return {
            'decision_speed': {
                'average': avg_decision_time,
                'consistency': decision_consistency
            },
            'risk_patterns': risk_patterns
        }

    def _analyze_bias_patterns(self, history: List[Dict], trades: List[Dict]) -> Dict:
        """Analyze trading bias patterns"""
        # Extract bias indicators
        entry_biases = [trade.get('entry_bias', 'neutral') for trade in trades]
        exit_biases = [trade.get('exit_bias', 'neutral') for trade in trades]
        
        # Analyze common biases
        bias_counts = {
            'confirmation_bias': sum(1 for b in entry_biases if b == 'confirmation'),
            'anchoring_bias': sum(1 for b in entry_biases if b == 'anchoring'),
            'loss_aversion': sum(1 for b in exit_biases if b == 'loss_aversion'),
            'recency_bias': sum(1 for b in entry_biases if b == 'recency')
        }
        
        # Calculate bias severity
        total_trades = len(trades) if trades else 1
        bias_severity = {bias: count/total_trades 
                        for bias, count in bias_counts.items()}
        
        return {
            'bias_counts': bias_counts,
            'bias_severity': bias_severity
        }

    def _detect_cycles(self, values: List[float]) -> Dict:
        """Detect cycles in numerical sequences"""
        if not values:
            return {'cycle_length': 0, 'cycle_strength': 0}
            
        # Use autocorrelation to detect cycles
        autocorr = np.correlate(values, values, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        # Find peaks in autocorrelation
        peaks = [i for i in range(1, len(autocorr)-1) 
                if autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1]]
        
        if peaks:
            cycle_length = peaks[0]
            cycle_strength = autocorr[peaks[0]] / autocorr[0]
        else:
            cycle_length = 0
            cycle_strength = 0
            
        return {
            'cycle_length': cycle_length,
            'cycle_strength': cycle_strength
        }

    def _detect_trends(self, values: List[float]) -> Dict:
        """Detect trends in numerical sequences"""
        if not values:
            return {'trend_direction': 'neutral', 'trend_strength': 0}
            
        # Calculate trend using linear regression
        x = np.arange(len(values))
        slope, _ = np.polyfit(x, values, 1)
        
        # Determine trend direction and strength
        trend_direction = 'up' if slope > 0.01 else 'down' if slope < -0.01 else 'neutral'
        trend_strength = abs(slope)
        
        return {
            'trend_direction': trend_direction,
            'trend_strength': trend_strength
        }

    def _generate_recommendations(self, patterns: Dict) -> List[Dict]:
        """Generate trading recommendations based on psychological patterns"""
        recommendations = []
        
        # Check emotional patterns
        if patterns['emotional_patterns']['stability'] > 0.3:
            recommendations.append({
                'type': 'emotional',
                'message': 'High emotional volatility detected. Consider reducing position sizes.',
                'severity': 'high'
            })
            
        # Check confidence patterns
        if patterns['confidence_patterns']['extremes']['high_confidence'] > 3:
            recommendations.append({
                'type': 'confidence',
                'message': 'Potential overconfidence detected. Review risk management.',
                'severity': 'medium'
            })
            
        # Check bias patterns
        for bias, severity in patterns['bias_patterns']['bias_severity'].items():
            if severity > 0.3:
                recommendations.append({
                    'type': 'bias',
                    'message': f'High {bias} detected. Consider alternative perspectives.',
                    'severity': 'medium'
                })
                
        return recommendations

