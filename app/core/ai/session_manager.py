import logging
from typing import Dict, Optional
from app.core.ai.market_psychology import PsychologyPatternAnalyzer
from app.core.ai.technical_analyzer import TechnicalPatternAnalyzer
from app.core.ai.trading_session import TradingSession
from app.core.ai.zone_analyzer import ZonePatternAnalyzer
import pandas as pd
import numpy as np
from datetime import datetime

from app.core.domain.models.session_models import SessionStats
from app.core.utils.traning_logger import EnhancedTrainingLogger

class SessionManager:
    """Manages multiple trading sessions and extracts learning patterns"""
    
    def __init__(self, max_sessions: int = 10, log_dir: str = "logs/"):
        self.max_sessions = max_sessions
        self.sessions = []
        self.learned_patterns = {
            'psychological': {},
            'technical': {},
            'zone_based': {}
        }
        self.session_metrics = []
        self.performance_metrics = {}
        
        # Initialize pattern analyzers
        self.pattern_analyzers = {
            'psychology': PsychologyPatternAnalyzer(),
            'technical': TechnicalPatternAnalyzer(),
            'zone': ZonePatternAnalyzer()
        }
        
        # Initialize enhanced logger
        self.logger = logging.getLogger(__name__)
        self.session_logger = EnhancedTrainingLogger(log_dir)

    def create_new_session(self, 
                          market_data: pd.DataFrame, 
                          initial_conditions: Dict) -> TradingSession:
        """Create and initialize a new trading session"""
        # Apply learned patterns to initial conditions
        adjusted_conditions = self._apply_learned_patterns(initial_conditions)
        
        # Create new session
        session = TradingSession()
        session.initialize_session(
            initial_psychology=adjusted_conditions['psychology'],  # Changed from psychology to initial_psychology
            market_data=market_data,
            zones=adjusted_conditions['zones']
        )
        
        # Add to sessions list
        self.sessions.append(session)
        if len(self.sessions) > self.max_sessions:
            self._archive_oldest_session()
        
        return session
    
    def _apply_learned_patterns(self, conditions: Dict) -> Dict:
        """Apply learned patterns to new session conditions"""
        adjusted = conditions.copy()
        
        # Apply psychological adjustments
        if self.learned_patterns['psychological']:
            adjusted['psychology'] = self._adjust_psychology(
                conditions['psychology'],
                self.learned_patterns['psychological']
            )
        
        # Apply technical adjustments
        if self.learned_patterns['technical']:
            adjusted['technical'] = self._adjust_technical(
                conditions['technical'],
                self.learned_patterns['technical']
            )
        
        # Apply zone adjustments
        if self.learned_patterns['zone_based']:
            adjusted['zones'] = self._adjust_zones(
                conditions['zones'],
                self.learned_patterns['zone_based']
            )
        
        return adjusted
    
    def analyze_session_results(self, session: TradingSession) -> Dict:
        """Analyze completed session and extract patterns"""
        try:
            analysis = {
                'psychological_patterns': self.pattern_analyzers['psychology'].analyze(
                    session.state_history,
                    session.trades
                ),
                'technical_patterns': self.pattern_analyzers['technical'].analyze(
                    session.state_history,
                    session.trades
                ),
                'zone_patterns': self.pattern_analyzers['zone'].analyze(
                    session.state_history,
                    session.trades
                )
            }
            
            # Update learned patterns
            self._update_learned_patterns(analysis)
            
            # Log analysis using enhanced logger
            analysis_session_id = f"analysis_{datetime.now().timestamp()}"
            self.session_logger.start_session(analysis_session_id, 100000.0)
            self.session_logger.end_session(
                final_balance=session.final_balance or 100000.0,
                performance_metrics=analysis,
                psychological_state=session.psychological_state,
                technical_state=session.technical_state
            )
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing session results: {str(e)}")
            raise
    
    def _update_learned_patterns(self, new_analysis: Dict):
        """Update learned patterns with new session analysis"""
        for category in ['psychological', 'technical', 'zone_based']:
            if category not in self.learned_patterns:
                self.learned_patterns[category] = {}
            
            # Update each pattern type
            for pattern_type, pattern_data in new_analysis[f'{category}_patterns'].items():
                if pattern_type not in self.learned_patterns[category]:
                    self.learned_patterns[category][pattern_type] = pattern_data
                else:
                    # Weighted average with existing patterns (more weight to recent)
                    self.learned_patterns[category][pattern_type] = {
                        'pattern': self._weighted_average(
                            self.learned_patterns[category][pattern_type]['pattern'],
                            pattern_data['pattern'],
                            0.7  # 70% weight to existing patterns
                        ),
                        'confidence': self._weighted_average(
                            self.learned_patterns[category][pattern_type]['confidence'],
                            pattern_data['confidence'],
                            0.7
                        )
                    }
    
    def _weighted_average(self, existing: float, new: float, weight: float) -> float:
        """Calculate weighted average between existing and new values"""
        return existing * weight + new * (1 - weight)
    
    def get_session_recommendations(self, 
                                  current_market_state: Dict,
                                  psychological_state: Dict) -> Dict:
        """
        Generate comprehensive trading recommendations based on current market and psychological states
        
        Args:
            current_market_state: Current market technical and zone state
            psychological_state: Current psychological patterns and state
        
        Returns:
            Dict containing recommendations for different aspects of trading
        """
        try:
            recommendations = {
                'psychological_adjustments': self._get_psychological_recommendations(psychological_state),
                'technical_setups': self._get_technical_recommendations(current_market_state),
                'zone_strategies': self._get_zone_recommendations(current_market_state),
                'risk_adjustments': self._get_risk_recommendations(
                    psychological_state,
                    current_market_state
                )
            }
            
            # Log recommendations using enhanced logger
            rec_session_id = f"recommendations_{datetime.now().timestamp()}"
            self.session_logger.start_session(rec_session_id, 100000.0)
            self.session_logger.end_session(
                final_balance=100000.0,
                performance_metrics=recommendations,
                psychological_state=psychological_state,
                technical_state=current_market_state
            )
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {str(e)}")
            raise
    
    def _get_psychological_recommendations(self, psych_state: Dict) -> Dict:
        """Generate psychological recommendations"""
        recommendations = {}
        
        # Check emotional state
        emotional_balance = psych_state.get('emotional_balance', 0.5)
        if emotional_balance > 0.7:
            recommendations['emotional'] = {
                'action': 'reduce_risk',
                'reason': 'High emotional state detected - potential overconfidence',
                'adjustment': 'Consider reducing position sizes by 25%'
            }
        elif emotional_balance < 0.3:
            recommendations['emotional'] = {
                'action': 'increase_focus',
                'reason': 'Low emotional state detected - potential fear',
                'adjustment': 'Review trading plan and stick to predefined rules'
            }
        
        # Check confidence levels
        confidence = psych_state.get('confidence', 0.5)
        if confidence < 0.3:
            recommendations['confidence'] = {
                'action': 'rebuild_confidence',
                'reason': 'Low confidence detected',
                'adjustment': 'Take smaller positions with higher probability setups'
            }
        
        return recommendations
    
    def _get_technical_recommendations(self, market_state: Dict) -> Dict:
        """Generate technical trading recommendations"""
        recommendations = {}
        
        # Apply learned technical patterns
        for pattern, data in self.learned_patterns['technical'].items():
            if self._pattern_matches(pattern, market_state):
                recommendations[pattern] = {
                    'setup': data['pattern'],
                    'confidence': data['confidence'],
                    'suggested_action': self._get_suggested_action(pattern, data)
                }
        
        return recommendations
    
    def _get_zone_recommendations(self, market_state: Dict) -> Dict:
        """Generate zone-based trading recommendations"""
        recommendations = {}
        
        # Check zone patterns
        for pattern, data in self.learned_patterns['zone_based'].items():
            if self._zone_pattern_matches(pattern, market_state):
                recommendations[pattern] = {
                    'zone_type': data['pattern']['type'],
                    'strength': data['pattern']['strength'],
                    'suggested_strategy': self._get_zone_strategy(data)
                }
        
        return recommendations
    
    def _get_risk_recommendations(self, 
                                psych_state: Dict,
                                market_state: Dict) -> Dict:
        """Generate risk management recommendations"""
        # Base risk level
        base_risk = 0.02  # 2% base risk per trade
        
        # Adjust based on psychological state
        psych_adjustment = self._calculate_psychological_risk_adjustment(psych_state)
        
        # Adjust based on market conditions
        market_adjustment = self._calculate_market_risk_adjustment(market_state)
        
        # Calculate final risk parameters
        adjusted_risk = base_risk * psych_adjustment * market_adjustment
        
        return {
            'position_size': {
                'base_risk': base_risk,
                'adjusted_risk': adjusted_risk,
                'psychological_factor': psych_adjustment,
                'market_factor': market_adjustment
            },
            'stop_loss': self._calculate_stop_loss_recommendations(
                market_state,
                adjusted_risk
            ),
            'take_profit': self._calculate_take_profit_recommendations(
                market_state,
                adjusted_risk
            )
        }
    
    def _calculate_psychological_risk_adjustment(self, psych_state: Dict) -> float:
        """Calculate risk adjustment based on psychological state"""
        confidence = psych_state.get('confidence', 0.5)
        emotional_balance = psych_state.get('emotional_balance', 0.5)
        
        # Reduce risk for extreme psychological states
        confidence_factor = np.clip(confidence, 0.5, 1.0)
        emotional_factor = 1 - abs(emotional_balance - 0.5)
        
        return confidence_factor * emotional_factor
    
    def _calculate_market_risk_adjustment(self, market_state: Dict) -> float:
        """Calculate risk adjustment based on market conditions"""
        volatility = market_state.get('volatility', 0.5)
        trend_strength = market_state.get('trend_strength', 0.5)
        
        # Reduce risk in high volatility, increase in strong trends
        volatility_factor = 1 - np.clip(volatility - 0.5, 0, 0.5)
        trend_factor = 1 + np.clip(trend_strength - 0.5, 0, 0.5)
        
        return volatility_factor * trend_factor
    
    def archive_and_learn(self):
        """Archive session data and update learning patterns"""
        if not self.sessions:
            return
        
        # Analyze all active sessions
        for session in self.sessions:
            analysis = self.analyze_session_results(session)
            self.session_metrics.append(analysis)
        
        # Extract and update patterns
        self._extract_cross_session_patterns()
        self._update_performance_metrics()
        
        # Archive sessions
        self._archive_sessions()
    
    def _extract_cross_session_patterns(self):
        """Extract patterns across multiple sessions"""
        # Implement cross-session pattern recognition
        pass
    
    def _archive_sessions(self):
        """Archive completed sessions"""
        # Implement session archival logic
        pass

    def _generate_recommendation_summary(self,
                                       psych_recs: Dict,
                                       tech_recs: Dict,
                                       zone_recs: Dict,
                                       risk_recs: Dict) -> Dict:
        """Generate a summary of all recommendations"""
        summary = {
            'primary_focus': self._determine_primary_focus(
                psych_recs,
                tech_recs,
                zone_recs
            ),
            'risk_stance': self._determine_risk_stance(risk_recs),
            'key_actions': []
        }
        
        # Add psychological actions
        if 'emotional' in psych_recs:
            summary['key_actions'].append(psych_recs['emotional']['action'])
        
        # Add technical actions
        for setup in tech_recs.values():
            if setup.get('confidence', 0) > 0.7:
                summary['key_actions'].append(setup['suggested_action'])
        
        # Add zone actions
        for strategy in zone_recs.values():
            if strategy.get('strength', 0) > 0.7:
                summary['key_actions'].append(strategy['suggested_strategy'])
        
        return summary

    def _generate_session_advice(self,
                               market_state: Dict,
                               psych_state: Dict,
                               recommendations: Dict) -> Dict:
        """Generate session-specific trading advice"""
        # Reference TradingSession._generate_trading_advice implementation
        confidence = psych_state.get('confidence_patterns', {}).get('current', 0.5)
        emotional_balance = psych_state.get('emotional_patterns', {}).get('stability', 0.5)
        trend_strength = market_state.get('technical_state', {}).get('trend', {}).get('strength', 0.5)
        
        state_assessment = {
            'confidence': 'high' if confidence > 0.7 else 'low' if confidence < 0.3 else 'moderate',
            'emotional': 'balanced' if 0.4 <= emotional_balance <= 0.6 else 'extreme',
            'trend': 'strong' if trend_strength > 0.7 else 'weak' if trend_strength < 0.3 else 'moderate'
        }
        
        return {
            'state_assessment': state_assessment,
            'trade_frequency': self._get_trade_frequency_advice(state_assessment),
            'position_sizing': self._get_position_sizing_advice(
                state_assessment,
                recommendations['risk_adjustments']
            ),
            'focus_areas': self._get_focus_areas(state_assessment, market_state)
        }

    def _generate_session_insights(self) -> Dict:
        """Generate insights from current session state"""
        return {
            'session_state': {
                'interval': self.current_interval,
                'psychological': {
                    'confidence': self.psychological_state.get('confidence', 0.5),
                    'emotional_balance': self.psychological_state.get('emotional_balance', 0.5)
                },
                'technical': {
                    'trend_direction': self.technical_state['trend']['direction'],
                    'trend_strength': self.technical_state['trend']['strength']
                },
                'zones': {
                    'in_support': self.zone_state['in_support_zone'],
                    'in_resistance': self.zone_state['in_resistance_zone']
                }
            },
            'trading_advice': self._generate_trading_advice(),
            'risk_advice': self._generate_risk_advice(),
            'performance': self._calculate_performance_metrics()
        }
    
    def _generate_trading_advice(self) -> Dict:
        """Generate trading advice based on current state"""
        confidence = self.psychological_state.get('confidence', 0.5)
        emotional_balance = self.psychological_state.get('emotional_balance', 0.5)
        trend_strength = self.technical_state['trend']['strength']
        
        # Base state assessment
        state_assessment = {
            'confidence': 'high' if confidence > 0.7 else 'low' if confidence < 0.3 else 'moderate',
            'emotional': 'balanced' if 0.4 <= emotional_balance <= 0.6 else 'extreme',
            'trend': 'strong' if trend_strength > 0.7 else 'weak' if trend_strength < 0.3 else 'moderate'
        }
        
        # Generate advice
        advice = {
            'entry_conditions': self._get_entry_conditions(),
            'position_sizing': self._get_position_sizing_advice(),
            'psychological_adjustment': self._get_psychological_adjustment()
        }
        
        return {
            'state_assessment': state_assessment,
            'advice': advice
        }

    def save_session_stats(self, session_stats: SessionStats) -> None:
        """Save session statistics and maintain session limit"""
        try:
            # Convert session stats to dictionary
            stats_dict = {
                'session_id': session_stats.session_id,
                'start_time': session_stats.start_time.isoformat(),
                'end_time': session_stats.end_time.isoformat(),
                'total_trades': session_stats.total_trades,
                'winning_trades': session_stats.winning_trades,
                'losing_trades': session_stats.losing_trades,
                'win_rate': float(session_stats.win_rate),
                'avg_profit': float(session_stats.avg_profit),
                'max_drawdown': float(session_stats.max_drawdown),
                'sharpe_ratio': float(session_stats.sharpe_ratio),
                'psychological_state': session_stats.psychological_state,
                'technical_state': session_stats.technical_state
            }
            
            # Add session to list
            self.sessions.append(stats_dict)
            
            # Maintain maximum sessions limit
            if len(self.sessions) > self.max_sessions:
                self._remove_worst_performing_session()
                
            # Update performance metrics
            self._update_performance_metrics()
            
            # Log session using enhanced logger
            self.session_logger.start_session(session_stats.session_id, 100000.0)
            self.session_logger.end_session(
                final_balance=100000.0 * (1 + session_stats.avg_profit),
                performance_metrics={
                    'win_rate': session_stats.win_rate,
                    'avg_profit': session_stats.avg_profit,
                    'max_drawdown': session_stats.max_drawdown,
                    'sharpe_ratio': session_stats.sharpe_ratio
                },
                psychological_state=session_stats.psychological_state,
                technical_state=session_stats.technical_state
            )
            
        except Exception as e:
            self.logger.error(f"Error saving session stats: {str(e)}")
            raise

    def _remove_worst_performing_session(self):
        """Remove worst performing session based on combined metric"""
        self.sessions.sort(
            key=lambda x: (
                x['sharpe_ratio'] * 0.4 +
                x['win_rate'] * 0.3 -
                x['max_drawdown'] * 0.3
            ),
            reverse=True
        )
        self.sessions.pop()

    def _update_performance_metrics(self) -> None:
        """Update aggregate performance metrics"""
        try:
            if not self.sessions:
                return
                
            metrics = {
                'avg_win_rate': np.mean([s['win_rate'] for s in self.sessions]),
                'avg_profit': np.mean([s['avg_profit'] for s in self.sessions]),
                'avg_sharpe': np.mean([s['sharpe_ratio'] for s in self.sessions]),
                'best_session': max(self.sessions, key=lambda x: x['sharpe_ratio'])['session_id'],
                'worst_session': min(self.sessions, key=lambda x: x['sharpe_ratio'])['session_id'],
                'total_trades': sum(s['total_trades'] for s in self.sessions),
                'winning_trades': sum(s['winning_trades'] for s in self.sessions),
                'losing_trades': sum(s['losing_trades'] for s in self.sessions)
            }
            
            self.performance_metrics = metrics
            
        except Exception as e:
            self.logger.error(f"Error updating performance metrics: {str(e)}")
            
    def rank_sessions(self) -> None:
        """Rank sessions based on performance metrics"""
        try:
            self.sessions.sort(
                key=lambda x: x['sharpe_ratio'] - x['max_drawdown'],
                reverse=True
            )
            
        except Exception as e:
            self.logger.error(f"Error ranking sessions: {str(e)}")
            
    def maintain_min_sessions(self, min_sessions: int) -> None:
        """Ensure minimum number of sessions is maintained"""
        try:
            while len(self.sessions) > min_sessions:
                worst_session = min(
                    self.sessions,
                    key=lambda x: x['sharpe_ratio'] - x['max_drawdown']
                )
                self.sessions.remove(worst_session)
                
                # Log session removal
                self.session_logger.log_forecast(
                    predicted_values=[],
                    confidence=0.0,
                    technical_indicators={},
                    actual_value=None
                )
                
            self._update_performance_metrics()
            
        except Exception as e:
            self.logger.error(f"Error maintaining minimum sessions: {str(e)}")
            
    def get_best_session(self) -> Optional[Dict]:
        """Get the best performing session"""
        try:
            if not self.sessions:
                return None
                
            return max(
                self.sessions,
                key=lambda x: x['sharpe_ratio'] - x['max_drawdown']
            )
            
        except Exception as e:
            self.logger.error(f"Error getting best session: {str(e)}")
            return None
            
    def get_session_stats(self, session_id: str) -> Optional[Dict]:
        """Get statistics for a specific session"""
        try:
            for session in self.sessions:
                if session['session_id'] == session_id:
                    return session
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting session stats: {str(e)}")
            return None
            
    def get_performance_summary(self) -> Dict:
        """Get summary of overall performance metrics"""
        try:
            return {
                'total_sessions': len(self.sessions),
                'performance_metrics': self.performance_metrics,
                'recent_sessions': self.sessions[-5:] if self.sessions else []
            }
            
        except Exception as e:
            self.logger.error(f"Error getting performance summary: {str(e)}")
            return {}







