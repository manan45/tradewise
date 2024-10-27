import logging


class TradingPsychology:
    """Analyzes and manages trading psychology based on market conditions"""
    
    def __init__(self):
        self.risk_tolerance = 0.5  # Base risk tolerance (0-1)
        self.confidence_level = 0.5  # Base confidence level (0-1)
        self.emotional_state = 0.5  # Neutral emotional state (0-1, fear to greed)
        self.zone_threshold = 0.10  # 10% zone threshold
        
        # Psychological bias weights
        self.bias_weights = {
            'trend_following': 0.3,
            'mean_reversion': 0.2,
            'support_resistance': 0.2,
            'momentum': 0.15,
            'volume': 0.15
        }
    
    def analyze_trader_psychology(self, current_data: pd.DataFrame, zones: dict) -> Dict[str, float]:
        """
        Analyze current trading psychology based on market conditions
        """
        try:
            current_price = float(current_data['close'].iloc[-1])
            
            # Calculate psychological metrics
            psychological_state = {
                'risk_tolerance': self._calculate_risk_tolerance(current_data),
                'confidence': self._calculate_confidence_level(current_data, zones),
                'emotional_balance': self._calculate_emotional_state(current_data),
                'stress_level': self._calculate_stress_level(current_data),
                'decision_quality': self._calculate_decision_quality(current_data),
                'bias_exposure': self._calculate_psychological_biases(current_data, zones)
            }
            
            # Add zone-based psychological adjustments
            zone_psychology = self._analyze_zone_psychology(current_price, zones)
            psychological_state.update(zone_psychology)
            
            return psychological_state
            
        except Exception as e:
            logging.error(f"Error in psychological analysis: {str(e)}")
            return {}

    def _calculate_risk_tolerance(self, df: pd.DataFrame) -> float:
        """Calculate current risk tolerance based on market conditions"""
        try:
            # Recent volatility impact
            volatility = df['volatility'].iloc[-1]
            vol_ma = df['volatility'].rolling(20).mean().iloc[-1]
            vol_impact = 1 - min(1, volatility / vol_ma)  # Lower volatility = higher tolerance
            
            # Recent performance impact
            returns = df['close'].pct_change().tail(10)
            win_rate = (returns > 0).mean()
            performance_impact = win_rate
            
            # Trend strength impact
            trend_strength = abs(df['trend_strength'].iloc[-1])
            
            risk_tolerance = (
                self.risk_tolerance * 0.4 +  # Base level
                vol_impact * 0.3 +           # Volatility impact
                performance_impact * 0.2 +    # Recent performance
                trend_strength * 0.1          # Trend impact
            )
            
            return float(np.clip(risk_tolerance, 0.1, 0.9))
            
        except Exception as e:
            logging.error(f"Error calculating risk tolerance: {str(e)}")
            return self.risk_tolerance

    def _calculate_confidence_level(self, df: pd.DataFrame, zones: dict) -> float:
        """Calculate trading confidence based on technical and psychological factors"""
        try:
            # Technical confirmation
            rsi = float(df['rsi'].iloc[-1])
            macd_hist = float(df['macd_hist'].iloc[-1])
            bb_position = float(df['bb_position'].iloc[-1])
            
            # Zone confirmation
            price = float(df['close'].iloc[-1])
            nearest_support = max([z for z in zones['support_zones'] if z < price], default=price*0.9)
            nearest_resistance = min([z for z in zones['resistance_zones'] if z > price], default=price*1.1)
            
            zone_confidence = min(
                abs(price - nearest_support) / price,
                abs(price - nearest_resistance) / price
            )
            
            # Trend confirmation
            trend_aligned = (
                (price > df['sma_20'].iloc[-1] and df['sma_20'].iloc[-1] > df['sma_50'].iloc[-1]) or
                (price < df['sma_20'].iloc[-1] and df['sma_20'].iloc[-1] < df['sma_50'].iloc[-1])
            )
            
            confidence = (
                self.confidence_level * 0.3 +                    # Base confidence
                (0.5 + trend_aligned * 0.5) * 0.3 +             # Trend alignment
                (1 - zone_confidence) * 0.2 +                    # Zone proximity
                (abs(50 - rsi) / 50) * 0.1 +                    # RSI extremes
                (np.sign(macd_hist) * bb_position) * 0.1        # MACD & BB confirmation
            )
            
            return float(np.clip(confidence, 0.1, 0.9))
            
        except Exception as e:
            logging.error(f"Error calculating confidence: {str(e)}")
            return self.confidence_level

    def _calculate_emotional_state(self, df: pd.DataFrame) -> float:
        """Calculate emotional state (fear to greed scale)"""
        try:
            # Price momentum
            returns = df['close'].pct_change()
            momentum = returns.tail(5).mean()
            
            # Volatility state
            current_vol = df['volatility'].iloc[-1]
            vol_ma = df['volatility'].rolling(20).mean().iloc[-1]
            vol_state = current_vol / vol_ma
            
            # RSI extremes
            rsi = df['rsi'].iloc[-1]
            rsi_factor = (rsi - 50) / 50  # -1 to 1 scale
            
            # Volume pressure
            volume_ratio = df['volume'].iloc[-1] / df['volume'].rolling(20).mean().iloc[-1]
            
            emotional_state = (
                self.emotional_state * 0.3 +      # Base emotional state
                np.sign(momentum) * 0.2 +         # Price momentum impact
                (1 - vol_state) * 0.2 +           # Volatility impact
                rsi_factor * 0.2 +                # RSI impact
                (volume_ratio - 1) * 0.1          # Volume impact
            )
            
            return float(np.clip(emotional_state, 0, 1))
            
        except Exception as e:
            logging.error(f"Error calculating emotional state: {str(e)}")
            return self.emotional_state

    def _analyze_zone_psychology(self, price: float, zones: dict) -> Dict[str, Any]:
        """Analyze psychological impact of price zones"""
        try:
            # Find nearest zones
            supports = sorted(zones['support_zones'])
            resistances = sorted(zones['resistance_zones'])
            
            # Zone analysis
            zone_status = {
                'in_support_zone': any(abs(price - s)/price <= self.zone_threshold for s in supports),
                'in_resistance_zone': any(abs(price - r)/price <= self.zone_threshold for r in resistances),
                'zone_strength': 0.0,
                'zone_pressure': 0.0
            }
            
            # Calculate zone strength
            if zone_status['in_support_zone']:
                nearest_support = max([s for s in supports if s < price], default=price*0.9)
                zone_status['zone_strength'] = 1 - abs(price - nearest_support)/(price * self.zone_threshold)
                zone_status['zone_pressure'] = 1  # Upward pressure
                
            elif zone_status['in_resistance_zone']:
                nearest_resistance = min([r for r in resistances if r > price], default=price*1.1)
                zone_status['zone_strength'] = 1 - abs(price - nearest_resistance)/(price * self.zone_threshold)
                zone_status['zone_pressure'] = -1  # Downward pressure
            
            return zone_status
            
        except Exception as e:
            logging.error(f"Error in zone psychology analysis: {str(e)}")
            return {'in_support_zone': False, 'in_resistance_zone': False, 'zone_strength': 0.0, 'zone_pressure': 0.0}

    def _calculate_psychological_biases(self, df: pd.DataFrame, zones: dict) -> Dict[str, float]:
        """Calculate exposure to common psychological biases"""
        try:
            current_price = float(df['close'].iloc[-1])
            
            biases = {
                'anchoring_bias': self._calculate_anchoring_bias(df),
                'confirmation_bias': self._calculate_confirmation_bias(df),
                'loss_aversion': self._calculate_loss_aversion(df),
                'overconfidence': self._calculate_overconfidence(df, zones),
                'recency_bias': self._calculate_recency_bias(df)
            }
            
            return biases
            
        except Exception as e:
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



