import os
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import json
import pandas as pd
import numpy as np
from dataclasses import dataclass

from app.core.ai.technical_analyzer import TechnicalPatternAnalyzer
from app.core.ai.market_environment import MarketEnvironment
from app.core.ai.zone_analyzer import ZonePatternAnalyzer
from app.core.ai.market_psychology import PsychologyPatternAnalyzer
from app.core.ai.trading_psychology import TradingPsychology
from app.core.ai.model_builder import build_lstm_model
from app.core.ai.session_manager import SessionManager

@dataclass
class SessionStats:
    """Statistics for a trading session"""
    session_id: str
    start_time: datetime
    end_time: datetime
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_profit: float
    max_drawdown: float
    sharpe_ratio: float
    psychological_state: Dict
    technical_state: Dict

@dataclass
class PredictionStats:
    """Statistics for prediction accuracy"""
    session_id: str
    timestamp: datetime
    mae: float
    mse: float
    accuracy: float
    precision: float
    recall: float
    predictions: List[Dict]
    actual_values: List[float]

class TradewiseAI:
    def __init__(self, 
                 model_path: str = "./models/",
                 session_save_dir: str = "sessions/",
                 log_dir: str = "logs/",
                 min_sessions: int = 50):
        """Initialize TradewiseAI"""
        self.model_path = model_path
        self.session_save_dir = session_save_dir
        self.log_dir = log_dir
        
        # Initialize components
        self.psychology_analyzer = PsychologyPatternAnalyzer()
        self.trading_psychology = TradingPsychology()
        self.session_manager = SessionManager()
        self.technical_analyzer = TechnicalPatternAnalyzer()
        self.zone_analyzer = ZonePatternAnalyzer()
        
        # Setup logging
        self._setup_logging()
        
        self.session_manager = SessionManager()
        # Create necessary directories
        os.makedirs(model_path, exist_ok=True)
        os.makedirs(session_save_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        self.min_sessions = min_sessions
        self.look_back = 60  # Number of time steps to look back
        self.model = None
        self.model_config = {}
        self.model_weights = None

    def _setup_logging(self):
        """Setup logging configuration"""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # File handler
        log_filename = os.path.join(self.log_dir, f'tradewise_{datetime.now().strftime("%Y%m%d")}.log')
        fh = logging.FileHandler(log_filename)
        fh.setLevel(logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    async def generate_trade_suggestions(self, data_feed: pd.DataFrame) -> List[Dict]:
        """Generate trade suggestions based on market analysis and psychological factors"""
        try:
            self.logger.info("Generating trade suggestions")
            
            # Calculate technical indicators
            df_indicators = self._prepare_technical_indicators(data_feed)
            
            # Get current market state
            current_data = df_indicators  # Use all available data for analysis
            
            # Get market analysis
            market_analysis = self._analyze_market(current_data)
            
            # Analyze psychological factors
            psych_analysis = self.psychology_analyzer.analyze(
                market_analysis['state_history'],
                market_analysis['trades']
            )
            
            # Analyze zones
            zone_analysis = self.zone_analyzer.analyze(
                market_analysis['state_history'],
                market_analysis['trades']
            )
            
            # Get session recommendations
            session_recommendations = self.session_manager.get_session_recommendations(
                market_analysis['current_state'],
                psych_analysis['patterns']
            )
            
            # Generate trading signals
            signals = self._generate_trading_signals(
                current_data,
                market_analysis,
                psych_analysis,
                zone_analysis
            )
            
            # Format suggestions with comprehensive analysis
            suggestions = []
            for signal in signals:
                suggestion = {
                    'action': signal['action'],
                    'entry': {
                        'price': float(signal['entry_price']),
                        'confidence': float(signal['confidence']),
                        'timeframe': signal['timeframe']
                    },
                    'risk_management': {
                        'stop_loss': float(signal['stop_loss']),
                        'take_profit': float(signal['take_profit']),
                        'position_size': float(signal['position_size']),
                        'risk_reward_ratio': float(signal['risk_reward_ratio'])
                    },
                    'analysis': {
                        'technical': market_analysis['current_state']['technical_state'],
                        'psychological': psych_analysis['patterns'],
                        'zones': zone_analysis['patterns']
                    },
                    'recommendations': session_recommendations,
                    'rationale': self._generate_trade_rationale(
                        signal,
                        market_analysis,
                        psych_analysis,
                        zone_analysis
                    )
                }
                suggestions.append(suggestion)
            
            # Log session stats
            self._log_session_stats({
                'suggestions': suggestions,
                'market_state': market_analysis['current_state'],
                'psychological_state': psych_analysis['patterns'],
                'timestamp': datetime.now()
            })
            
            return suggestions
            
        except Exception as e:
            self.logger.error(f"Error generating trade suggestions: {str(e)}")
            raise

    async def get_session_stats(self, session_id: Optional[str] = None) -> List[SessionStats]:
        """Get statistics for trading sessions"""
        try:
            if session_id:
                stats_file = os.path.join(self.session_save_dir, session_id, 'stats.json')
                if os.path.exists(stats_file):
                    with open(stats_file, 'r') as f:
                        return SessionStats(**json.load(f))
                return None
            
            # Get all session stats
            stats = []
            for session_dir in os.listdir(self.session_save_dir):
                stats_file = os.path.join(self.session_save_dir, session_dir, 'stats.json')
                if os.path.exists(stats_file):
                    with open(stats_file, 'r') as f:
                        stats.append(SessionStats(**json.load(f)))
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting session stats: {str(e)}")
            return []

    async def get_prediction_stats(self, session_id: Optional[str] = None) -> List[PredictionStats]:
        """Get statistics for predictions"""
        try:
            if session_id:
                pred_file = os.path.join(self.session_save_dir, session_id, 'predictions.json')
                if os.path.exists(pred_file):
                    with open(pred_file, 'r') as f:
                        return PredictionStats(**json.load(f))
                return None
            
            # Get all prediction stats
            stats = []
            for session_dir in os.listdir(self.session_save_dir):
                pred_file = os.path.join(self.session_save_dir, session_dir, 'predictions.json')
                if os.path.exists(pred_file):
                    with open(pred_file, 'r') as f:
                        stats.append(PredictionStats(**json.load(f)))
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting prediction stats: {str(e)}")
            return []

    async def get_session_logs(self, session_id: Optional[str] = None, 
                             start_date: Optional[str] = None,
                             end_date: Optional[str] = None) -> List[Dict]:
        """Get logs for trading sessions"""
        try:
            logs = []
            log_files = []
            
            if session_id:
                # Get logs for specific session
                log_file = os.path.join(self.log_dir, f'tradewise_{session_id}.log')
                if os.path.exists(log_file):
                    log_files.append(log_file)
            else:
                # Get all log files within date range
                start_dt = datetime.strptime(start_date, '%Y-%m-%d') if start_date else None
                end_dt = datetime.strptime(end_date, '%Y-%m-%d') if end_date else None
                
                for file in os.listdir(self.log_dir):
                    if file.startswith('tradewise_') and file.endswith('.log'):
                        file_date = datetime.strptime(file.split('_')[1].split('.')[0], '%Y%m%d')
                        if (not start_dt or file_date >= start_dt) and (not end_dt or file_date <= end_dt):
                            log_files.append(os.path.join(self.log_dir, file))
            
            # Parse log files
            for log_file in log_files:
                with open(log_file, 'r') as f:
                    for line in f:
                        try:
                            timestamp, name, level, message = line.strip().split(' - ')
                            logs.append({
                                'timestamp': timestamp,
                                'name': name,
                                'level': level,
                                'message': message
                            })
                        except:
                            continue
            
            return sorted(logs, key=lambda x: x['timestamp'], reverse=True)
            
        except Exception as e:
            self.logger.error(f"Error getting session logs: {str(e)}")
            return []

    def _analyze_market(self, df: pd.DataFrame) -> Dict:
        """Analyze market conditions and generate state history"""
        try:
            # Get last 120 periods for analysis
            current_data = df[-120:]
            
            # Initialize state history
            state_history = []
            trades = []  # Will be populated if trade data is available
            
            # Calculate market states for each period
            for i in range(len(current_data)):
                window = current_data[:i+1]
                
                technical_state = self._calculate_technical_state(window)
                psychological_state = self._calculate_psychological_state(window)
                zone_state = self._calculate_zone_state(window)
                
                state = {
                    'timestamp': window.index[-1],
                    'technical_state': technical_state,
                    'psychological_state': psychological_state,
                    'zone_state': zone_state
                }
                state_history.append(state)
            
            return {
                'state_history': state_history,
                'trades': trades,
                'current_state': state_history[-1]
            }
            
        except Exception as e:
            self.logger.error(f"Error in market analysis: {str(e)}")
            raise

    def _calculate_technical_state(self, df: pd.DataFrame) -> Dict:
        """Calculate technical indicators and state"""
        try:
            latest = df.iloc[-1]
            
            return {
                'close': float(latest['close']),
                'high': float(latest['high']),
                'low': float(latest['low']),
                'volume': float(latest['volume']),
                'trend': {
                    'direction': 'up' if latest['close'] > df['close'].mean() else 'down', # todo this can be improved
                    'strength': self._calculate_trend_strength(df),
                    'consistency': self._calculate_trend_consistency(df)
                },
                'momentum': {
                    'rsi': float(latest.get('rsi', 50)),
                    'macd': float(latest.get('macd', 0)),
                    'macd_signal': float(latest.get('macd_signal', 0)),
                    'macd_hist': float(latest.get('macd_hist', 0))
                },
                'volatility': float(latest.get('volatility', df['close'].std())),
                'volume_trend': self._calculate_volume_trend(df)
            }
        except Exception as e:
            self.logger.error(f"Error calculating technical state: {str(e)}")
            return {}

    def _calculate_psychological_state(self, df: pd.DataFrame) -> Dict:
        """Calculate psychological state based on market behavior"""
        try:
            # Get current zones
            zones = self.zone_analyzer.identify_zone(df)
            
            # Calculate psychological metrics
            psychological_state = self.trading_psychology.analyze_trader_psychology(df, zones)
            
            # Add market psychology metrics
            market_psychology = self.trading_psychology.analyze_market_psychology(df)
            psychological_state.update(market_psychology)
            
            return psychological_state
            
        except Exception as e:
            self.logger.error(f"Error calculating psychological state: {str(e)}")
            return {}

    def _calculate_zone_state(self, df: pd.DataFrame) -> Dict:
        """Calculate price zone state"""
        try:
            current_price = float(df['close'].iloc[-1])
            
            # Identify key zones
            zones = self.price_analyzer.identify_zones(df)
            
            # Calculate zone metrics
            zone_state = {
                'support_zones': zones['support_zones'],
                'resistance_zones': zones['resistance_zones'],
                'current_zone': self._identify_current_zone(current_price, zones),
                'zone_strength': self._calculate_zone_strength(current_price, zones),
                'breakout_potential': self._calculate_breakout_potential(df, zones)
            }
            
            return zone_state
            
        except Exception as e:
            self.logger.error(f"Error calculating zone state: {str(e)}")
            return {}

    def _generate_predictions(self, df: pd.DataFrame) -> Dict:
        """Generate price predictions and confidence levels"""
        try:
            # Prepare features for prediction
            features = self._prepare_prediction_features(df)
            
            # Generate predictions for different timeframes
            predictions = {
                'short_term': self._predict_timeframe(features, timeframe='short'),
                'medium_term': self._predict_timeframe(features, timeframe='medium'),
                'long_term': self._predict_timeframe(features, timeframe='long')
            }
            
            # Calculate prediction metrics
            metrics = self._calculate_prediction_metrics(predictions, df)
            
            return {
                'predictions': predictions,
                'metrics': metrics,
                'confidence': self._calculate_prediction_confidence(metrics)
            }
            
        except Exception as e:
            self.logger.error(f"Error generating predictions: {str(e)}")
            return {}

    def _prepare_prediction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for prediction models"""
        try:
            features = pd.DataFrame()
            
            # Technical features
            features['trend_strength'] = self._calculate_trend_strength(df)
            features['volatility'] = df['close'].rolling(window=20).std()
            features['volume_trend'] = self._calculate_volume_trend(df)
            
            # Momentum features
            features['rsi'] = df.get('rsi', pd.Series([50] * len(df)))
            features['macd_hist'] = df.get('macd_hist', pd.Series([0] * len(df)))
            
            # Zone features
            zones = self.zone_analyzer.identify_zone(df)
            features['zone_strength'] = self._calculate_zone_strength(df['close'], zones)
            
            return features.fillna(method='ffill')
            
        except Exception as e:
            self.logger.error(f"Error preparing prediction features: {str(e)}")
            return pd.DataFrame()
            
    def _calculate_zone_strength(self, df: pd.DataFrame) -> pd.Series:
        """Calculate the strength of the price trend"""
        try:
            # Calculate short and long term moving averages
            short_ma = df['close'].rolling(window=20).mean()
            long_ma = df['close'].rolling(window=50).mean()
            
            # Calculate trend strength based on MA crossovers and slope
            trend_strength = (short_ma - long_ma) / long_ma
            
            # Normalize between -1 and 1
            return trend_strength.clip(-1, 1)
            
        except Exception as e:
            self.logger.error(f"Error calculating trend strength: {str(e)}")
            return pd.Series([0] * len(df))
            
    def _calculate_volume_trend(self, df: pd.DataFrame) -> pd.Series:
        """Calculate the trend in trading volume"""
        try:
            # Calculate volume moving average
            vol_ma = df['volume'].rolling(window=20).mean()
            
            # Compare current volume to moving average
            volume_trend = (df['volume'] - vol_ma) / vol_ma
            
            # Normalize between -1 and 1
            return volume_trend.clip(-1, 1)
            
        except Exception as e:
            self.logger.error(f"Error calculating volume trend: {str(e)}")
            return pd.Series([0] * len(df))

    def _format_suggestions(self, predictions: Dict, 
                          market_analysis: Dict,
                          psych_analysis: Dict,
                          zone_analysis: Dict) -> List[Dict]:
        """Format trading suggestions with comprehensive analysis"""
        try:
            current_price = float(market_analysis['current_state']['technical_state']['close'])
            
            suggestions = []
            
            # Generate suggestions for different timeframes
            for timeframe, prediction in predictions['predictions'].items():
                if self._should_generate_suggestion(prediction, market_analysis):
                    suggestion = {
                        'timeframe': timeframe,
                        'action': self._determine_action(prediction, current_price),
                        'entry_price': self._calculate_entry_price(prediction, current_price),
                        'stop_loss': self._calculate_stop_loss(prediction, zone_analysis),
                        'take_profit': self._calculate_take_profit(prediction, zone_analysis),
                        'confidence': float(prediction['confidence']),
                        'analysis': {
                            'technical': market_analysis['current_state']['technical_state'],
                            'psychological': psych_analysis['patterns'],
                            'zones': zone_analysis['patterns']
                        },
                        'rationale': self._generate_suggestion_rationale(
                            prediction,
                            market_analysis,
                            psych_analysis,
                            zone_analysis
                        )
                    }
                    suggestions.append(suggestion)
            
            return suggestions
            
        except Exception as e:
            self.logger.error(f"Error formatting suggestions: {str(e)}")
            return

    def _should_generate_suggestion(self, prediction: Dict, market_analysis: Dict) -> bool:
        """Determine if a suggestion should be generated based on confidence and conditions"""
        try:
            # Check prediction confidence
            if prediction['confidence'] < 0.6:  # Minimum confidence threshold
                return False
                
            # Check market conditions
            technical_state = market_analysis['current_state']['technical_state']
            
            # Don't generate suggestions in extremely volatile conditions
            if technical_state['volatility'] > 0.8:  # High volatility threshold
                return False
                
            # Check trend strength
            if technical_state['trend']['strength'] < 0.3:  # Weak trend threshold
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error in suggestion validation: {str(e)}")
            return False

    def _generate_suggestion_rationale(self, prediction: Dict,
                                     market_analysis: Dict,
                                     psych_analysis: Dict,
                                     zone_analysis: Dict) -> Dict:
        """Generate detailed rationale for trading suggestion"""
        try:
            technical_state = market_analysis['current_state']['technical_state']
            
            rationale = {
                'technical_factors': {
                    'trend': {
                        'direction': technical_state['trend']['direction'],
                        'strength': technical_state['trend']['strength'],
                        'analysis': self._analyze_trend_context(technical_state)
                    },
                    'momentum': {
                        'rsi': technical_state['momentum']['rsi'],
                        'macd': technical_state['momentum']['macd_hist'],
                        'analysis': self._analyze_momentum_context(technical_state)
                    }
                },
                'psychological_factors': {
                    'market_psychology': psych_analysis['patterns']['emotional_patterns'],
                    'trader_psychology': psych_analysis['patterns']['decision_patterns'],
                    'recommendations': psych_analysis['recommendations']
                },
                'zone_factors': {
                    'current_zone': zone_analysis['patterns']['current_zone'],
                    'zone_strength': zone_analysis['strength'],
                    'breakout_potential': zone_analysis['patterns']['breakout_zones']
                }
            }
            
            return rationale
            
        except Exception as e:
            self.logger.error(f"Error generating suggestion rationale: {str(e)}")
            return {}

    def _log_session_stats(self, stats: Dict):
        """Log session statistics"""
        try:
            session_stats = SessionStats(
                session_id=str(datetime.now().timestamp()),
                start_time=stats['timestamp'],
                end_time=datetime.now(),
                total_trades=len(stats['suggestions']),
                winning_trades=0,  # To be updated after trade completion
                losing_trades=0,   # To be updated after trade completion
                win_rate=0.0,      # To be updated after trade completion
                avg_profit=0.0,    # To be updated after trade completion
                max_drawdown=0.0,  # To be calculated from historical data
                sharpe_ratio=0.0,  # To be calculated from historical data
                psychological_state=stats['psychological_state'],
                technical_state=stats['market_state']
            )
            
            # Save session stats
            self.session_manager.save_session_stats(session_stats)
            
        except Exception as e:
            self.logger.error(f"Error logging session stats: {str(e)}")

    def _prepare_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for analysis"""
        try:
            return self.technical_analyzer.calculate_all_indicators(df)
        except Exception as e:
            self.logger.error(f"Error calculating technical indicators: {str(e)}")
            return df

    def _generate_trading_signals(self,
                                current_data: pd.DataFrame,
                                market_analysis: Dict,
                                psych_analysis: Dict,
                                zone_analysis: Dict) -> List[Dict]:
        """Generate trading signals based on all analyses"""
        try:
            signals = []
            current_price = float(current_data['close'].iloc[-1])
            
            # Get technical signals
            tech_signals = self.technical_analyzer.generate_signals(market_analysis)
            
            # Get psychological adjustments
            psych_adjustments = self.trading_psychology.get_psychological_advice(
                psych_analysis['patterns']
            )
            
            # Get zone signals
            zone_signals = self.zone_analyzer.generate_signals(zone_analysis)
            
            # Combine and filter signals
            for signal in tech_signals:
                if self._validate_signal(signal, psych_adjustments, zone_signals):
                    adjusted_signal = self._adjust_signal_parameters(
                        signal,
                        psych_adjustments,
                        zone_signals,
                        current_price
                    )
                    signals.append(adjusted_signal)
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error generating trading signals: {str(e)}")
            return []

    def _generate_trade_rationale(self,
                                signal: Dict,
                                market_analysis: Dict,
                                psych_analysis: Dict,
                                zone_analysis: Dict) -> Dict:
        """Generate detailed rationale for trade suggestion"""
        try:
            return {
                'technical_factors': self._analyze_technical_factors(
                    signal,
                    market_analysis
                ),
                'psychological_factors': self._analyze_psychological_factors(
                    signal,
                    psych_analysis
                ),
                'zone_factors': self._analyze_zone_factors(
                    signal,
                    zone_analysis
                ),
                'risk_factors': self._analyze_risk_factors(signal)
            }
        except Exception as e:
            self.logger.error(f"Error generating trade rationale: {str(e)}")
            return {}

    def _validate_signal(self, signal: Dict, psych_adjustments: Dict, zone_signals: List[Dict]) -> bool:
        """Validate trading signal based on psychological adjustments and zone signals"""
        try:
            # Check if signal is valid based on psychological adjustments
            if signal['action'] not in psych_adjustments['patterns']:
                return False
            
            # Check if signal is valid based on zone signals
            for zone_signal in zone_signals:
                if zone_signal['action'] == signal['action'] and zone_signal['timeframe'] == signal['timeframe']:
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error validating signal: {str(e)}")
            return False

    def _adjust_signal_parameters(self, signal: Dict, psych_adjustments: Dict, zone_signals: List[Dict], current_price: float) -> Dict:
        """Adjust signal parameters based on psychological adjustments and zone signals"""
        try:
            # Get psychological adjustment for the signal
            adjustment = psych_adjustments['patterns'][signal['action']]
            
            # Get zone signal for the signal
            zone_signal = next((zone for zone in zone_signals if zone['action'] == signal['action'] and zone['timeframe'] == signal['timeframe']), None)
            
            # Adjust signal parameters
            signal['entry_price'] = float(signal['entry_price']) + adjustment['adjustment']
            signal['stop_loss'] = float(signal['stop_loss']) + adjustment['stop_loss']
            signal['take_profit'] = float(signal['take_profit']) + adjustment['take_profit']
            signal['position_size'] = float(signal['position_size']) * (1 + adjustment['position_size'])
            signal['risk_reward_ratio'] = float(signal['risk_reward_ratio']) * (1 + adjustment['risk_reward_ratio'])
            
            # Calculate new stop loss and take profit based on current price
            signal['stop_loss'] = self._calculate_stop_loss(signal, zone_signal, current_price)
            signal['take_profit'] = self._calculate_take_profit(signal, zone_signal, current_price)
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Error adjusting signal parameters: {str(e)}")
            return {}

    def _calculate_stop_loss(self, signal: Dict, zone_signal: Dict, current_price: float) -> float:
        """Calculate stop loss based on zone signal and current price"""
        try:
            # Get zone signal for the signal
            zone_signal = next((zone for zone in zone_signals if zone['action'] == signal['action'] and zone['timeframe'] == signal['timeframe']), None)
            
            # Calculate stop loss based on zone signal and current price
            stop_loss = current_price - zone_signal['stop_loss']
            
            return stop_loss
            
        except Exception as e:
            self.logger.error(f"Error calculating stop loss: {str(e)}")
            return 0.0

    def _calculate_take_profit(self, signal: Dict, zone_signal: Dict, current_price: float) -> float:
        """Calculate take profit based on zone signal and current price"""
        try:
            # Get zone signal for the signal
            zone_signal = next((zone for zone in zone_signals if zone['action'] == signal['action'] and zone['timeframe'] == signal['timeframe']), None)
            
            # Calculate take profit based on zone signal and current price
            take_profit = current_price + zone_signal['take_profit']
            
            return take_profit
            
        except Exception as e:
            self.logger.error(f"Error calculating take profit: {str(e)}")
            return 0.0

    def _analyze_technical_factors(self, signal: Dict, market_analysis: Dict) -> Dict:
        """Analyze technical factors for trading signal"""
        try:
            # Get technical state for the signal
            technical_state = next((state for state in market_analysis['current_state']['technical_state'] if state['action'] == signal['action'] and state['timeframe'] == signal['timeframe']), None)
            
            # Analyze technical factors
            technical_factors = {
                'trend': {
                    'direction': technical_state['trend']['direction'],
                    'strength': technical_state['trend']['strength'],
                    'analysis': self._analyze_trend_context(technical_state)
                },
                'momentum': {
                    'rsi': technical_state['momentum']['rsi'],
                    'macd': technical_state['momentum']['macd_hist'],
                    'analysis': self._analyze_momentum_context(technical_state)
                }
            }
            
            return technical_factors
            
        except Exception as e:
            self.logger.error(f"Error analyzing technical factors: {str(e)}")
            return {}

    def _analyze_psychological_factors(self, signal: Dict, psych_analysis: Dict) -> Dict:
        """Analyze psychological factors for trading signal"""
        try:
            # Get psychological state for the signal
            psychological_state = next((state for state in psych_analysis['patterns'] if state['action'] == signal['action']), None)
            
            # Analyze psychological factors
            psychological_factors = {
                'market_psychology': psychological_state['emotional_patterns'],
                'trader_psychology': psychological_state['decision_patterns'],
                'recommendations': psych_analysis['recommendations']
            }
            
            return psychological_factors
            
        except Exception as e:
            self.logger.error(f"Error analyzing psychological factors: {str(e)}")
            return {}

    def _analyze_zone_factors(self, signal: Dict, zone_analysis: Dict) -> Dict:
        """Analyze zone factors for trading signal"""
        try:
            # Get zone state for the signal
            zone_state = next((state for state in zone_analysis['patterns'] if state['action'] == signal['action'] and state['timeframe'] == signal['timeframe']), None)
            
            # Analyze zone factors
            zone_factors = {
                'current_zone': zone_state['current_zone'],
                'zone_strength': zone_state['zone_strength'],
                'breakout_potential': zone_state['breakout_potential']
            }
            
            return zone_factors
            
        except Exception as e:
            self.logger.error(f"Error analyzing zone factors: {str(e)}")
            return {}

    def _analyze_risk_factors(self, signal: Dict) -> Dict:
        """Analyze risk factors for trading signal"""
        try:
            # Calculate risk factors
            risk_factors = {
                'position_size': float(signal['position_size']),
                'risk_reward_ratio': float(signal['risk_reward_ratio'])
            }
            
            return risk_factors
            
        except Exception as e:
            self.logger.error(f"Error analyzing risk factors: {str(e)}")
            return {}

    def _analyze_trend_context(self, technical_state: Dict) -> Dict:
        """Analyze trend context for trading signal"""
        try:
            # Analyze trend context
            trend_context = {
                'direction': technical_state['trend']['direction'],
                'strength': technical_state['trend']['strength'],
                'consistency': technical_state['trend']['consistency']
            }
            
            return trend_context
            
        except Exception as e:
            self.logger.error(f"Error analyzing trend context: {str(e)}")
            return {}

    def _analyze_momentum_context(self, technical_state: Dict) -> Dict:
        """Analyze momentum context for trading signal"""
        try:
            # Analyze momentum context
            momentum_context = {
                'rsi': technical_state['momentum']['rsi'],
                'macd': technical_state['momentum']['macd_hist'],
                'analysis': technical_state['momentum']['analysis']
            }
            
            return momentum_context
            
        except Exception as e:
            self.logger.error(f"Error analyzing momentum context: {str(e)}")
            return {}

    def _calculate_trend_strength(self, df: pd.DataFrame) -> float:
        """Calculate the strength of the price trend"""
        try:
            # Calculate short and long term moving averages
            short_ma = df['close'].rolling(window=20).mean()
            long_ma = df['close'].rolling(window=50).mean()
            
            # Calculate trend strength based on MA crossovers and slope
            trend_strength = (short_ma - long_ma) / long_ma
            
            # Normalize between -1 and 1
            return trend_strength.clip(-1, 1)
            
        except Exception as e:
            self.logger.error(f"Error calculating trend strength: {str(e)}")
            return 0.0
    def _calculate_trend_consistency(self, df: pd.DataFrame) -> float:
        """Calculate the consistency of the price trend"""
        try:
            # Calculate short and long term moving averages
            short_ma = df['close'].rolling(window=20).mean()
            long_ma = df['close'].rolling(window=50).mean()            
            # Calculate trend consistency based on MA crossovers and slope
            trend_consistency = (short_ma - long_ma) / long_ma
            
            # Normalize between -1 and 1
            return trend_consistency.clip(-1, 1)
        except Exception as e:
            self.logger.error(f"Error calculating trend consistency: {str(e)}")
            return 0.0

    async def train(self, data_feed: pd.DataFrame):
        """Train the model using historical data"""
        try:
            self.logger.info("Starting training process")

            # Calculate data points for 7 days at 5-minute intervals
            total_points = 7 * 24 * 12  # 12 five-minute intervals per hour
            
            if len(data_feed) < total_points:
                raise ValueError("Insufficient data for training")

            # Split data into training and evaluation sets
            train_size = int(total_points * (6.5/7))  # Use 6.5 days for training
            train_data = data_feed.iloc[:train_size]
            eval_data = data_feed.iloc[train_size:train_size + 24]  # Next 2 hours for evaluation

            session_predictions = []
            
            # Create multiple training sessions
            for _ in range(self.min_sessions):
                # Initialize and train model
                model = self._initialize_model()
                await self._train_model(model, train_data)
                
                # Generate predictions
                predictions = await self._predict_future(model, train_data)
                
                # Evaluate predictions
                actual = eval_data['close'].values
                performance = self._evaluate_predictions(predictions, actual)
                
                # Create session stats
                session_id = str(datetime.now().timestamp())
                session_stats = SessionStats(
                    session_id=session_id,
                    start_time=datetime.now(),
                    end_time=datetime.now(),
                    total_trades=len(predictions),
                    winning_trades=sum(1 for p, a in zip(predictions, actual) if abs(p-a)/a < 0.01),
                    losing_trades=sum(1 for p, a in zip(predictions, actual) if abs(p-a)/a >= 0.01),
                    win_rate=performance['accuracy'],
                    avg_profit=performance['avg_profit'],
                    max_drawdown=performance['max_drawdown'],
                    sharpe_ratio=performance['sharpe_ratio'],
                    psychological_state=self.trading_psychology.analyze_trader_psychology(
                        train_data, 
                        self.zone_analyzer.identify_zone(train_data['close'].values)
                    ),
                    technical_state=self._calculate_technical_state(train_data)
                )
                
                self.session_manager.save_session_stats(session_stats)
                session_predictions.append({
                    'session_id': session_id,
                    'model': model,
                    'performance': performance
                })

            # Rank sessions and maintain minimum count
            self.session_manager.rank_sessions()
            self.session_manager.maintain_min_sessions(self.min_sessions)
            
            # Reinforce model using best performing sessions
            await self._reinforce_model(session_predictions)
            
            self.logger.info("Training completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error during training: {str(e)}")
            raise

    async def _train_model(self, model, train_data: pd.DataFrame):
        """Train the model on historical data"""
        try:
            # Prepare features
            features = self._prepare_prediction_features(train_data)
            X, y = self._prepare_sequences(features, train_data['close'])
            
            # Train model
            model.fit(
                X, y,
                epochs=50,
                batch_size=32,
                validation_split=0.2,
                verbose=0
            )
            
            # Save weights
            self.model_weights = model.get_weights()
            
        except Exception as e:
            self.logger.error(f"Error training model: {str(e)}")
            raise

    def _initialize_model(self):
        """Initialize the LSTM model"""
        try:
            input_shape = (self.look_back, self._get_feature_count())
            model = build_lstm_model(input_shape)
            
            if self.model_weights is not None:
                model.set_weights(self.model_weights)
                
            return model
            
        except Exception as e:
            self.logger.error(f"Error initializing model: {str(e)}")
            raise

    def _get_feature_count(self) -> int:
        """Get the number of features used in the model"""
        # Count of features used in _prepare_prediction_features
        return 6  # Adjust based on actual feature count

    async def _predict_future(self, model, data: pd.DataFrame, steps: int = 24) -> List[float]:
        """Generate predictions for future time steps"""
        try:
            # Prepare latest data
            features = self._prepare_prediction_features(data)
            last_sequence = features.iloc[-self.look_back:].values
            
            predictions = []
            current_sequence = last_sequence.copy()
            
            # Generate predictions for each future step
            for _ in range(steps):
                # Reshape sequence for prediction
                X = current_sequence.reshape(1, self.look_back, -1)
                
                # Generate prediction
                pred = model.predict(X, verbose=0)[0][0]
                predictions.append(pred)
                
                # Update sequence for next prediction
                current_sequence = np.roll(current_sequence, -1, axis=0)
                current_sequence[-1] = self._update_features(current_sequence[-2], pred)
                
            return predictions
            
        except Exception as e:
            self.logger.error(f"Error generating predictions: {str(e)}")
            raise

    def _prepare_sequences(self, features: pd.DataFrame, target: pd.Series):
        """Prepare sequences for LSTM training"""
        X, y = [], []
        
        for i in range(len(features) - self.look_back):
            X.append(features.iloc[i:i + self.look_back].values)
            y.append(target.iloc[i + self.look_back])
            
        return np.array(X), np.array(y)

    def _evaluate_predictions(self, predictions: List[float], actual: np.ndarray) -> Dict:
        """Evaluate prediction performance"""
        try:
            predictions = np.array(predictions[:len(actual)])
            
            # Calculate metrics
            mse = np.mean((predictions - actual) ** 2)
            mae = np.mean(np.abs(predictions - actual))
            accuracy = np.mean(np.abs(predictions - actual) / actual < 0.01)
            
            # Calculate returns
            pred_returns = np.diff(predictions) / predictions[:-1]
            actual_returns = np.diff(actual) / actual[:-1]
            
            # Calculate Sharpe ratio
            excess_returns = pred_returns - 0.02/252  # Assuming 2% risk-free rate
            sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) if len(excess_returns) > 0 else 0
            
            # Calculate max drawdown
            cumulative_returns = np.cumprod(1 + pred_returns)
            max_drawdown = np.min(cumulative_returns / np.maximum.accumulate(cumulative_returns) - 1)
            
            return {
                'mse': float(mse),
                'mae': float(mae),
                'accuracy': float(accuracy),
                'sharpe_ratio': float(sharpe_ratio),
                'max_drawdown': float(max_drawdown),
                'avg_profit': float(np.mean(pred_returns))
            }
            
        except Exception as e:
            self.logger.error(f"Error evaluating predictions: {str(e)}")
            raise

    async def _reinforce_model(self, session_predictions: List[Dict]):
        """Reinforce model using best performing sessions"""
        try:
            # Sort sessions by performance
            sorted_sessions = sorted(
                session_predictions,
                key=lambda x: x['performance']['sharpe_ratio'] - x['performance']['max_drawdown'],
                reverse=True
            )
            
            # Get top performing session
            best_session = sorted_sessions[0]
            
            # Update model weights
            self.model_weights = best_session['model'].get_weights()
            
            # Save best model
            self._save_model(best_session['model'], best_session['session_id'])
            
        except Exception as e:
            self.logger.error(f"Error reinforcing model: {str(e)}")
            raise

    def _save_model(self, model, session_id: str):
        """Save model to disk"""
        try:
            model_dir = os.path.join(self.model_path, session_id)
            os.makedirs(model_dir, exist_ok=True)
            model.save(os.path.join(model_dir, 'model.h5'))
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            raise
