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
