import os
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from dataclasses import dataclass
from tensorflow.keras.models import Model

from app.core.ai.technical_analyzer import TechnicalPatternAnalyzer
from app.core.ai.market_environment import MarketEnvironment
from app.core.ai.trading_session import TradingSession
from app.core.ai.zone_analyzer import ZonePatternAnalyzer
from app.core.ai.market_psychology import PsychologyPatternAnalyzer
from app.core.ai.trading_psychology import TradingPsychology
from app.core.ai.model_builder import ModelBuilder, TrainingResult
from app.core.ai.session_manager import SessionManager
from app.core.ai.reinforcement.market_state import MarketState
from app.core.ai.reinforcement.action_space import ActionSpace
from app.core.ai.reinforcement.reward_calculator import RewardCalculator
from app.core.domain.models.trade_suggestion import DetailedTradeSuggestion
from app.core.domain.models.session_models import SessionStats

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
    model: Optional[object] = None
    reinforcement_stats: Optional[Dict] = None
    prediction_accuracy: Optional[float] = None

class TradewiseAI:
    def __init__(self, 
                 model_path: str = "./models/",
                 session_save_dir: str = "sessions/",
                 log_dir: str = "logs/",
                 min_sessions: int = 50,
                 max_sessions: int = 100):
        """Initialize TradewiseAI with session management"""
        self.model_path = model_path
        self.session_save_dir = session_save_dir
        self.log_dir = log_dir
        self.min_sessions = min_sessions
        self.max_sessions = max_sessions
        
        # Initialize components
        self.technical_analyzer = TechnicalPatternAnalyzer()
        self.market_environment = MarketEnvironment()
        self.zone_analyzer = ZonePatternAnalyzer()
        self.psychology_analyzer = PsychologyPatternAnalyzer()
        self.trading_psychology = TradingPsychology()
        
        # Initialize session manager for logging
        self.session_manager = SessionManager(max_sessions=max_sessions, log_dir=log_dir)
        
        self.model_builder = ModelBuilder()
        
        # Initialize market environment components
        self.market_state = MarketState()
        self.action_space = ActionSpace()
        self.reward_calculator = RewardCalculator()
        
        # RL parameters
        self.gamma = 0.95  # Discount factor
        self.epsilon = 0.1  # Exploration rate
        self.learning_rate = 0.001
        self.batch_size = 32
        
        # Setup logging and directories
        self.logger = logging.getLogger(__name__)
        self._create_directories()

    async def train(self, data_feed: pd.DataFrame) -> SessionStats:
        """Train model with multiple episodes and sessions, tracking best performers"""
        try:
            # Create training session ID
            training_id = f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.logger.info(f"Starting new training session: {training_id}")
            
            # Initialize training configuration
            training_config = {
                'data_length': len(data_feed),
                'feature_count': len(self.market_state.state_features),
                'gamma': self.gamma,
                'initial_epsilon': self.epsilon,
                'learning_rate': self.learning_rate
            }
            
            # Start training session in session manager's logger
            training_session_id = self.session_manager.session_logger.start_training_session(training_config)
            
            # Initialize market environment
            market_env = self._initialize_market_environment(data_feed)
            
            best_session = None
            best_performance = float('-inf')
            
            # Training parameters
            n_episodes = 15
            sessions_per_episode = 10
            max_steps = min(20, len(data_feed) - 1)
            
            try:
                for episode in range(n_episodes):
                    # Start new episode in session manager
                    episode_log = self.session_manager.session_logger.start_episode(episode)
                    
                    # Initialize episode model
                    input_shape = (market_env.state_size,)
                    episode_model = self.model_builder.build_rl_model(
                        input_shape,
                        market_env.action_space.n
                    )
                    
                    for session_num in range(sessions_per_episode):
                        session_id = f"{training_id}_e{episode}_s{session_num}"
                        
                        # Start new session in session manager
                        self.session_manager.session_logger.start_session(
                            session_id,
                            initial_balance=100000.0
                        )
                        
                        # Run training session
                        session_result = await self._run_training_session(
                            session_id,
                            episode_model,
                            market_env,
                            data_feed,
                            max_steps
                        )
                        
                        # Update best session if better performing
                        if session_result.sharpe_ratio > best_performance:
                            best_session = session_result
                            best_performance = session_result.sharpe_ratio
                        
                        # End session in session manager
                        self.session_manager.session_logger.end_session(
                            final_balance=session_result.final_balance,
                            performance_metrics={
                                'win_rate': session_result.win_rate,
                                'avg_profit': session_result.avg_profit,
                                'max_drawdown': session_result.max_drawdown,
                                'sharpe_ratio': session_result.sharpe_ratio
                            },
                            psychological_state=session_result.psychological_state,
                            technical_state=session_result.technical_state
                        )
                    
                    # End episode in session manager
                    self.session_manager.session_logger.end_episode()
                    
                    # Decay epsilon
                    self.epsilon = max(0.01, self.epsilon * 0.95)
                
                # End training session in session manager's logger
                self.session_manager.session_logger.end_training_session()
                
                # Save best session
                if best_session:
                    self.session_manager.save_session_stats(best_session)
                
                return best_session
                
            except Exception as e:
                # Ensure training session is ended even if there's an error
                self.session_manager.session_logger.end_training_session()
                raise e
            
        except Exception as e:
            self.logger.error(f"Error during training: {str(e)}")
            raise

    async def _run_training_session(self,
                                  session_id: str,
                                  model: Model,
                                  market_env: MarketEnvironment,
                                  data_feed: pd.DataFrame,
                                  max_steps: int) -> SessionStats:
        """Run a single training session with proper forecast handling"""
        try:
            # Create new trading session
            trading_session = self.session_manager.create_new_session(
                market_data=data_feed,
                initial_conditions={
                    'psychology': self._get_initial_psychological_state(),
                    'zones': self.zone_analyzer.identify_zone(data_feed['close'].values)
                }
            )
            
            # Initialize state and tracking variables
            state = market_env.reset()
            total_reward = 0.0
            session_trades = []
            session_start_time = datetime.now()
            
            for step in range(max_steps):
                # Ensure we have valid data indices
                if step >= len(data_feed):
                    break
                
                # Handle NaN states
                if np.any(np.isnan(state)):
                    state = np.nan_to_num(state, nan=0.0)
                
                # Generate forecast
                current_data = data_feed.iloc[step]
                forecast = self._generate_forecast(model, state, current_data)
                
                # Get action from forecast's best action
                action = int(forecast['best_action'])  # Ensure integer action
                
                # Take action
                next_state, reward, done, _, info = market_env.step(action)
                
                # Ensure valid reward
                reward = float(np.clip(np.nan_to_num(reward, 0.0), -1.0, 1.0))
                
                # Store trade if executed
                if info.get('trade_executed', False):
                    session_trades.append(info)
                
                # Train on experience
                loss = self._train_on_experience(
                    model,
                    state,
                    action,
                    reward,
                    next_state,
                    done
                )
                
                # Log reinforcement update
                self.session_manager.session_logger.log_reinforcement_update(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    q_values_before=forecast['q_values'],
                    q_values_after=model.predict_on_batch(next_state.reshape(1, -1))[0].tolist(),
                    epsilon=self.epsilon,
                    loss=loss
                )
                
                # Log forecast with safe index access
                next_price = data_feed.iloc[step + 1]['close'] if step + 1 < len(data_feed) else None
                self.session_manager.session_logger.log_forecast(
                    predicted_values=forecast['values'],
                    confidence=forecast['confidence'],
                    technical_indicators=forecast['technical_state'],
                    actual_value=next_price
                )
                
                # Update state and reward
                state = next_state
                total_reward += reward
                
                if done:
                    break
            
            # Calculate session metrics
            session_stats = SessionStats(
                session_id=session_id,
                start_time=session_start_time,
                end_time=datetime.now(),
                total_trades=len(session_trades),
                winning_trades=sum(1 for t in session_trades if t.get('trade_profit', 0) > 0),
                losing_trades=sum(1 for t in session_trades if t.get('trade_profit', 0) <= 0),
                win_rate=len([t for t in session_trades if t.get('trade_profit', 0) > 0]) / max(len(session_trades), 1),
                avg_profit=np.mean([t.get('trade_profit', 0) for t in session_trades]) if session_trades else 0.0,
                max_drawdown=max([abs(t.get('trade_profit', 0)) for t in session_trades]) if session_trades else 0.0,
                sharpe_ratio=self._calculate_sharpe_ratio(session_trades),
                psychological_state=trading_session.psychological_state,
                technical_state=trading_session.technical_state,
                reinforcement_stats={
                    'total_reward': float(total_reward),
                    'avg_reward': float(total_reward / max_steps),
                    'final_epsilon': float(self.epsilon)
                },
                final_balance=trading_session.final_balance or 100000.0
            )
            
            return session_stats
            
        except Exception as e:
            self.logger.error(f"Error running training session: {str(e)}")
            raise

    def _calculate_sharpe_ratio(self, trades: List[Dict]) -> float:
        """Calculate Sharpe ratio with safety checks"""
        try:
            if not trades:
                return 0.0
            
            returns = [float(t.get('trade_profit', 0)) for t in trades]
            if len(returns) < 2:
                return 0.0
            
            returns_std = float(np.std(returns))
            if returns_std == 0:
                return 0.0
            
            return float(np.mean(returns) / returns_std)
            
        except Exception as e:
            self.logger.error(f"Error calculating Sharpe ratio: {str(e)}")
            return 0.0

    def _initialize_market_environment(self, data_feed: pd.DataFrame) -> MarketEnvironment:
        """Initialize market environment with data feed"""
        try:
            # Calculate technical indicators
            technical_data = self.technical_analyzer.calculate_indicators(data_feed)
            
            # Initialize environment components
            market_state = MarketState()
            action_space = ActionSpace()
            reward_calculator = RewardCalculator()
            
            # Create environment
            market_env = MarketEnvironment(
                data=technical_data,
                action_space=action_space,
                market_state=market_state,
                reward_calculator=reward_calculator
            )
            
            return market_env
            
        except Exception as e:
            self.logger.error(f"Error initializing market environment: {str(e)}")
            raise

    def _train_on_experience(self,
                           model: Model,
                           state: np.ndarray,
                           action: int,
                           reward: float,
                           next_state: np.ndarray,
                           done: bool):
        """Train model on single experience tuple using ModelBuilder"""
        try:
            # Get current Q-values using predict_on_batch
            state_batch = state.reshape(1, -1)
            current_q = model.predict_on_batch(state_batch)[0]
            
            # Get next Q-values using predict_on_batch
            next_state_batch = next_state.reshape(1, -1)
            next_q = model.predict_on_batch(next_state_batch)[0]
            
            # Update Q-value for taken action
            if done:
                current_q[action] = reward
            else:
                current_q[action] = reward + self.gamma * np.max(next_q)
            
            # Train model using ModelBuilder's methods
            self.model_builder.train_on_batch(
                model,
                state_batch,
                current_q.reshape(1, -1)
            )
            
        except Exception as e:
            self.logger.error(f"Error training on experience: {str(e)}")
            raise

    def _update_session_stats(self,
                            session: SessionStats,
                            reward: float,
                            info: Dict):
        """Update session statistics based on step results"""
        try:
            # Initialize reinforcement_stats if not exists
            if not hasattr(session, 'reinforcement_stats') or session.reinforcement_stats is None:
                session.reinforcement_stats = {
                    'episode_rewards': [],
                    'total_rewards': 0,
                    'avg_reward': 0,
                    'max_reward': float('-inf'),
                    'min_reward': float('inf')
                }

            # Update trade counts
            if info is not None and info.get('trade_executed', False):
                session.total_trades += 1
                if info.get('trade_profit', 0) > 0:
                    session.winning_trades += 1
                else:
                    session.losing_trades += 1
            
            # Update win rate
            if session.total_trades > 0:
                session.win_rate = session.winning_trades / session.total_trades
            
            # Update profit metrics
            current_profit = info.get('trade_profit', 0) if info is not None else 0
            if session.total_trades > 0:
                session.avg_profit = (
                    (session.avg_profit * (session.total_trades - 1) + current_profit)
                    / session.total_trades
                )
            
            # Update drawdown
            if info is not None and info.get('drawdown', 0) > session.max_drawdown:
                session.max_drawdown = info['drawdown']
            
            # Update Sharpe ratio if we have enough rewards
            if len(session.reinforcement_stats['episode_rewards']) > 1:
                returns = np.array(session.reinforcement_stats['episode_rewards'])
                session.sharpe_ratio = (
                    np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
                )
            
            # Log the updates
            logger = logging.getLogger(f'session_{session.session_id}')
            logger.debug(
                f"Updated stats - Trades: {session.total_trades}, "
                f"Win Rate: {session.win_rate:.2%}, "
                f"Avg Profit: {session.avg_profit:.2%}"
            )
            
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Error updating session stats: {str(e)}")
            # Don't raise the exception to allow training to continue
            # but log it for debugging

    async def predict(self, data_feed: pd.DataFrame) -> List[Dict]:
        """Generate predictions using best performing session"""
        try:
            self.logger.info("Starting prediction process")
            
            # Get top performing sessions
            top_sessions = self.session_manager.get_top_sessions(n=3)
            if not top_sessions:
                raise ValueError("No trained sessions available")
            
            # Generate ensemble predictions
            predictions = []
            for session in top_sessions:
                pred = self._generate_session_predictions(session, data_feed)
                predictions.append(pred)
            
            # Combine predictions with weights based on session performance
            final_predictions = self._combine_predictions(predictions, top_sessions)
            
        except Exception as e:
            self.logger.error(f"Error preparing prediction features: {str(e)}")
            return pd.DataFrame()

    def _update_features(self, last_features: np.ndarray, prediction: float) -> np.ndarray:
        """Update feature vector with new prediction"""
        try:
            new_features = last_features.copy()
            
            # Update close price
            new_features[0] = prediction
            
            # Update returns
            new_features[1] = (prediction - last_features[0]) / last_features[0]
            
            # Update volatility (simple approximation)
            new_features[2] = abs(new_features[1])
            
            # Keep other features unchanged for now
            # In a more sophisticated implementation, you would update RSI, MACD, etc.
            
            return new_features
            
        except Exception as e:
            self.logger.error(f"Error updating features: {str(e)}")
            return last_features

    def _calculate_trend_strength(self, df: pd.DataFrame) -> float:
        """Calculate the strength of the price trend"""
        try:
            short_ma = df['close'].rolling(window=20).mean()
            long_ma = df['close'].rolling(window=50).mean()
            
            trend_strength = (short_ma - long_ma) / long_ma
            return float(trend_strength.iloc[-1].clip(-1, 1))
            
        except Exception as e:
            self.logger.error(f"Error calculating trend strength: {str(e)}")
            return 0.0

    def _calculate_trend_consistency(self, df: pd.DataFrame) -> float:
        """Calculate the consistency of the price trend"""
        try:
            short_ma = df['close'].rolling(window=20).mean()
            long_ma = df['close'].rolling(window=50).mean()            
            trend_consistency = (short_ma - long_ma) / long_ma
            
            return float(trend_consistency.iloc[-1].clip(-1, 1))
        except Exception as e:
            self.logger.error(f"Error calculating trend consistency: {str(e)}")
            return 0.0

    def _calculate_prediction_confidence(self,
                                      prediction: float,
                                      data_feed: pd.DataFrame,
                                      technical_state: Dict) -> float:
        """Calculate confidence score for a prediction"""
        try:
            # Technical analysis confidence
            technical_confidence = self._get_technical_confidence(
                prediction, data_feed, technical_state
            )
            
            # Psychological confidence
            psychological_confidence = self._get_psychological_confidence(
                prediction, data_feed
            )
            
            # Market state confidence
            market_confidence = self._get_market_confidence(
                prediction, data_feed
            )
            
            # Weighted average of confidence scores
            confidence = (
                technical_confidence * 0.4 +
                psychological_confidence * 0.3 +
                market_confidence * 0.3
            )
            
            return float(np.clip(confidence, 0, 1))
            
        except Exception as e:
            self.logger.error(f"Error calculating prediction confidence: {str(e)}")
            return 0.5

    def _get_psychological_state(self, training_result: TrainingResult) -> Dict:
        """Get psychological state from training result"""
        try:
            return {
                'confidence': training_result.metrics.get('confidence', 0.5),
                'emotional_balance': training_result.metrics.get('emotional_balance', 0.5),
                'risk_tolerance': training_result.metrics.get('risk_tolerance', 0.5),
                'stress_level': training_result.metrics.get('stress_level', 0.3)
            }
        except Exception as e:
            self.logger.error(f"Error getting psychological state: {str(e)}")
            return {}

    def _get_technical_state(self, training_result: TrainingResult) -> Dict:
        """Get technical state from training result"""
        try:
            return {
                'trend': {
                    'direction': training_result.metrics.get('trend_direction', 'neutral'),
                    'strength': training_result.metrics.get('trend_strength', 0.5)
                },
                'momentum': training_result.metrics.get('momentum', 0.0),
                'volatility': training_result.metrics.get('volatility', 0.0),
                'support_resistance': {
                    'support': training_result.metrics.get('support_level', 0.0),
                    'resistance': training_result.metrics.get('resistance_level', 0.0)
                }
            }
        except Exception as e:
            self.logger.error(f"Error getting technical state: {str(e)}")
            return {}

    def _get_technical_indicators(self, data_feed: pd.DataFrame, i: int) -> Dict:
        """Get technical indicators for a prediction"""
        try:
            window = data_feed.iloc[max(0, i-20):i+1]
            return {
                'rsi': self.technical_analyzer.calculate_rsi(window),
                'macd': self.technical_analyzer.calculate_macd(window),
                'bollinger': self.technical_analyzer.calculate_bollinger_bands(window),
                'volume_profile': self.technical_analyzer.analyze_volume_profile(window),
                'support_resistance': self.zone_analyzer.identify_zone(window['close'].values)
            }
        except Exception as e:
            self.logger.error(f"Error getting technical indicators: {str(e)}")
            return {}

    def _get_psychological_factors(self, data_feed: pd.DataFrame, i: int) -> Dict:
        """Get psychological factors for a prediction"""
        try:
            window = data_feed.iloc[max(0, i-20):i+1]
            psych_analysis = self.psychology_analyzer.analyze_market_psychology(window)
            return {
                'market_sentiment': psych_analysis.get('sentiment', 'neutral'),
                'fear_greed_index': psych_analysis.get('fear_greed', 50),
                'momentum_psychology': psych_analysis.get('momentum_psychology', 0.5),
                'volatility_sentiment': psych_analysis.get('volatility_sentiment', 0.5)
            }
        except Exception as e:
            self.logger.error(f"Error getting psychological factors: {str(e)}")
            return {}

    def _get_market_confidence(self, prediction: float, data_feed: pd.DataFrame) -> float:
        """Get market confidence for a prediction"""
        try:
            # Calculate recent price volatility
            returns = data_feed['close'].pct_change().dropna()
            volatility = returns.std()
            
            # Get market trend strength
            trend_strength = self.technical_analyzer.get_trend_strength(data_feed)
            
            # Get volume consistency
            volume_consistency = self.technical_analyzer.analyze_volume_consistency(data_feed)
            
            # Calculate confidence score
            confidence = (
                (1 - volatility) * 0.4 +  # Lower volatility = higher confidence
                trend_strength * 0.4 +     # Stronger trend = higher confidence
                volume_consistency * 0.2    # Consistent volume = higher confidence
            )
            
            return float(np.clip(confidence, 0, 1))
            
        except Exception as e:
            self.logger.error(f"Error calculating market confidence: {str(e)}")
            return 0.5

    async def generate_trade_suggestions(self, data_feed: pd.DataFrame) -> List[TradingSession]:
        """Generate trade suggestions from top performing sessions based on forecast confidence"""
        try:
            self.logger.info("Generating trade suggestions from sessions")
            
            if not self.session_manager.sessions:
                await self.train(data_feed)
            
            # Get all sessions and their predictions
            session_forecasts = []
            
            for session in self.session_manager.sessions:
                # Load model for this session
                model_path = os.path.join(self.model_path, session['session_id'], 'model.h5')
                model = self.model_builder.load_model(model_path)
                
                if model is None:
                    continue
                
                # Generate predictions
                predictions = await self.predict(data_feed)
                
                # Get analysis states
                technical_state = self._calculate_technical_state(data_feed)
                psych_state = self.trading_psychology.analyze_trader_psychology(
                    data_feed,
                    self.zone_analyzer.identify_zone(data_feed['close'].values)
                )
                
                # Determine action and confidence
                action, confidence = self._determine_trade_action(
                    predictions,
                    technical_state,
                    psych_state
                )
                
                # Calculate forecast metrics
                forecast_metrics = self._calculate_forecast_metrics(
                    predictions,
                    data_feed['close'].iloc[-1],
                    technical_state
                )
                
                session_forecasts.append({
                    'session': session,
                    'predictions': predictions,
                    'action': action,
                    'confidence': confidence,
                    'technical_state': technical_state,
                    'psych_state': psych_state,
                    'forecast_metrics': forecast_metrics
                })
            
            # Filter and sort sessions based on confidence and performance
            filtered_forecasts = [
                f for f in session_forecasts 
                if f['confidence'] > 0.6  # Minimum confidence threshold
                and f['session']['win_rate'] > 0.5  # Minimum win rate
                and f['session']['sharpe_ratio'] > 1.0  # Minimum Sharpe ratio
            ]
            
            # Sort by combined score (confidence + performance metrics)
            sorted_forecasts = sorted(
                filtered_forecasts,
                key=lambda x: (
                    x['confidence'] * 0.4 +  # 40% weight to confidence
                    x['session']['win_rate'] * 0.3 +  # 30% weight to win rate
                    (x['session']['sharpe_ratio'] / 3) * 0.3  # 30% weight to normalized Sharpe
                ),
                reverse=True
            )
            
            # Take top 20 forecasts
            top_forecasts = sorted_forecasts[:20]
            
            suggestions = []
            
            for forecast in top_forecasts:
                # Calculate risk parameters
                risk_params = self._calculate_risk_parameters(
                    forecast['predictions'],
                    data_feed['close'].iloc[-1],
                    forecast['technical_state']
                )
                
                # Create trade suggestion
                suggestion = DetailedTradeSuggestion(
                    Suggestion=f"Session {forecast['session']['session_id'][:8]} suggests {forecast['action']}",
                    Action=forecast['action'],
                    Summary={
                        "confidence": f"{forecast['confidence']:.2%}",
                        "session_performance": (
                            f"Win Rate: {forecast['session']['win_rate']:.2%}, "
                            f"Sharpe: {forecast['session']['sharpe_ratio']:.2f}"
                        ),
                        "predicted_movement": f"{forecast['forecast_metrics']['predicted_movement']:.2%}",
                        "forecast_strength": f"{forecast['forecast_metrics']['strength']:.2f}"
                    },
                    Risk_Management={
                        "stop_loss": f"{risk_params['stop_loss']:.2f}",
                        "take_profit": f"{risk_params['take_profit']:.2f}",
                        "position_size": f"{risk_params['position_size']:.2f}",
                        "risk_reward": f"{risk_params['risk_reward']:.2f}"
                    },
                    Technical_Analysis={
                        "trend": forecast['technical_state']['trend']['direction'],
                        "strength": f"{forecast['technical_state']['trend']['strength']:.2f}",
                        "momentum": (
                            f"RSI: {forecast['technical_state']['momentum']['rsi']:.2f}, "
                            f"MACD: {forecast['technical_state']['momentum']['macd']:.2f}"
                        )
                    },
                    Forecast_Time=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                )
                
                suggestions.append(suggestion)
            
            self.logger.info(f"Generated {len(suggestions)} filtered trade suggestions")
            return suggestions
            
        except Exception as e:
            self.logger.error(f"Error generating trade suggestions: {str(e)}")
            raise

    def _calculate_forecast_metrics(self, predictions: List[float], 
                                  current_price: float,
                                  technical_state: Dict) -> Dict:
        """Calculate forecast metrics and strength"""
        try:
            # Calculate predicted movement
            predicted_movement = (predictions[-1] - current_price) / current_price
            
            # Calculate forecast volatility
            forecast_volatility = np.std(predictions) / np.mean(predictions)
            
            # Calculate trend strength in predictions
            forecast_trend = np.polyfit(range(len(predictions)), predictions, 1)[0]
            trend_strength = abs(forecast_trend / current_price)
            
            # Calculate forecast consistency
            diffs = np.diff(predictions)
            consistency = np.mean(np.sign(diffs[1:]) == np.sign(diffs[:-1]))
            
            # Calculate overall forecast strength
            strength = (
                (1 - forecast_volatility) * 0.3 +  # Lower volatility is better
                trend_strength * 0.4 +            # Stronger trend is better
                consistency * 0.3                 # Higher consistency is better
            )
            
            return {
                'predicted_movement': predicted_movement,
                'volatility': forecast_volatility,
                'trend_strength': trend_strength,
                'consistency': consistency,
                'strength': strength
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating forecast metrics: {str(e)}")
            return {
                'predicted_movement': 0.0,
                'volatility': 0.0,
                'trend_strength': 0.0,
                'consistency': 0.0,
                'strength': 0.0
            }

    def _determine_trade_action(self, predictions: List[float], 
                              technical_state: Dict, 
                              psych_state: Dict) -> Tuple[str, float]:
        """Determine trade action and confidence level"""
        try:
            current_price = predictions[0]
            future_prices = predictions[1:]
            
            # Calculate price movement
            price_movement = (future_prices[-1] - current_price) / current_price
            
            # Calculate confidence based on multiple factors
            technical_confidence = technical_state['trend']['strength']
            psych_confidence = psych_state.get('confidence', 0.5)
            
            # Combined confidence
            confidence = (technical_confidence + psych_confidence) / 2
            
            # Determine action
            if price_movement > 0.01 and confidence > 0.6:
                action = "BUY"
            elif price_movement < -0.01 and confidence > 0.6:
                action = "SELL"
            else:
                action = "HOLD"
                
            return action, confidence
            
        except Exception as e:
            self.logger.error(f"Error determining trade action: {str(e)}")
            return "HOLD", 0.0

    def _calculate_risk_parameters(self, predictions: List[float], 
                                 current_price: float,
                                 technical_state: Dict) -> Dict:
        """Calculate risk management parameters"""
        try:
            # Calculate volatility
            volatility = technical_state['volatility']
            
            # Calculate stop loss and take profit distances based on volatility
            stop_distance = max(volatility * 2, 0.01)  # Minimum 1% stop loss
            profit_distance = max(volatility * 3, 0.02)  # Minimum 2% take profit
            
            # Calculate position size based on risk
            risk_per_trade = 0.02  # 2% risk per trade
            position_size = risk_per_trade / stop_distance
            
            # Calculate risk/reward ratio
            risk_reward = profit_distance / stop_distance
            
            return {
                'stop_loss': current_price * (1 - stop_distance),
                'take_profit': current_price * (1 + profit_distance),
                'position_size': position_size,
                'risk_reward': risk_reward
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating risk parameters: {str(e)}")
            return {
                'stop_loss': 0.0,
                'take_profit': 0.0,
                'position_size': 0.0,
                'risk_reward': 0.0
            }

    def _calculate_technical_state(self, df: pd.DataFrame) -> Dict:
        """Calculate current technical state of the market"""
        try:
            latest = df.iloc[-1]
            
            return {
                'trend': {
                    'direction': 'up' if latest['close'] > df['close'].mean() else 'down',
                    'strength': float(self._calculate_trend_strength(df)),
                    'consistency': float(self._calculate_trend_consistency(df))
                },
                'momentum': {
                    'rsi': float(latest.get('rsi', 50)),
                    'macd': float(latest.get('macd', 0)),
                    'macd_hist': float(latest.get('macd_hist', 0))
                },
                'volatility': float(latest['close'].rolling(window=20).std())
            }
        except Exception as e:
            self.logger.error(f"Error calculating technical state: {str(e)}")
            return {}

    def _prepare_prediction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for prediction"""
        try:
            features = pd.DataFrame()
            
            # Price-based features
            features['close'] = df['close']
            features['returns'] = df['close'].pct_change()
            features['volatility'] = df['close'].rolling(window=20).std()
            
            # Technical indicators
            features['rsi'] = df.get('rsi', pd.Series([50] * len(df)))
            features['macd'] = df.get('macd', pd.Series([0] * len(df)))
            features['trend_strength'] = self._calculate_trend_strength(df)
            
            return features.fillna(method='ffill')
            
        except Exception as e:
            self.logger.error(f"Error preparing prediction features: {str(e)}")
            return pd.DataFrame()


    def _calculate_trend_strength(self, df: pd.DataFrame) -> float:
        """Calculate the strength of the price trend"""
        try:
            short_ma = df['close'].rolling(window=20).mean()
            long_ma = df['close'].rolling(window=50).mean()
            
            trend_strength = (short_ma - long_ma) / long_ma
            return float(trend_strength.iloc[-1].clip(-1, 1))
            
        except Exception as e:
            self.logger.error(f"Error calculating trend strength: {str(e)}")
            return 0.0

    def _calculate_trend_consistency(self, df: pd.DataFrame) -> float:
        """Calculate the consistency of the price trend"""
        try:
            short_ma = df['close'].rolling(window=20).mean()
            long_ma = df['close'].rolling(window=50).mean()            
            trend_consistency = (short_ma - long_ma) / long_ma
            
            return float(trend_consistency.iloc[-1].clip(-1, 1))
        except Exception as e:
            self.logger.error(f"Error calculating trend consistency: {str(e)}")
            return 0.0

    def _get_initial_psychological_state(self) -> Dict:
        """Get initial psychological state"""
        try:
            return {
                'confidence': 0.5,
                'emotional_balance': 0.5,
                'risk_tolerance': 0.5,
                'stress_level': 0.3
            }
        except Exception as e:
            self.logger.error(f"Error getting initial psychological state: {str(e)}")
            return {}

    def _get_initial_technical_state(self) -> Dict:
        """Get initial technical state"""
        try:
            return {
                'trend': {
                    'direction': 'neutral',
                    'strength': 0.5
                },
                'momentum': 0.0,
                'volatility': 0.0,
                'support_resistance': {
                    'support': 0.0,
                    'resistance': 0.0
                }
            }
        except Exception as e:
            self.logger.error(f"Error getting initial technical state: {str(e)}")
            return {}

    def _setup_logging(self):
        """Setup logging configuration"""
        try:
            # Create logs directory if it doesn't exist
            os.makedirs(self.log_dir, exist_ok=True)
            
            # Configure logging
            log_file = os.path.join(self.log_dir, f'tradewise_{datetime.now().strftime("%Y%m%d")}.log')
            
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.FileHandler(log_file),
                    logging.StreamHandler()
                ]
            )
            
            self.logger = logging.getLogger(__name__)
            self.logger.info("Logging setup completed")
            
        except Exception as e:
            print(f"Error setting up logging: {str(e)}")
            raise

    def _create_directories(self):
        """Create necessary directories for models and sessions"""
        try:
            # Create model directory
            os.makedirs(self.model_path, exist_ok=True)
            self.logger.info(f"Model directory created/verified: {self.model_path}")
            
            # Create session directory
            os.makedirs(self.session_save_dir, exist_ok=True)
            self.logger.info(f"Session directory created/verified: {self.session_save_dir}")
            
            # Create subdirectories for different model types
            model_subdirs = ['lstm', 'reinforcement', 'ensemble']
            for subdir in model_subdirs:
                os.makedirs(os.path.join(self.model_path, subdir), exist_ok=True)
                
            self.logger.info("All required directories created/verified")
            
        except Exception as e:
            self.logger.error(f"Error creating directories: {str(e)}")
            raise

    def _load_existing_sessions(self):
        """Load existing session data from disk"""
        try:
            session_files = []
            
            # Get all session files
            for file in os.listdir(self.session_save_dir):
                if file.endswith('.json'):
                    session_files.append(os.path.join(self.session_save_dir, file))
            
            if not session_files:
                self.logger.info("No existing sessions found")
                return
            
            # Load each session file
            loaded_sessions = []
            for session_file in session_files:
                try:
                    with open(session_file, 'r') as f:
                        import json
                        session_data = json.load(f)
                        
                        # Convert string dates back to datetime
                        session_data['start_time'] = datetime.fromisoformat(session_data['start_time'])
                        session_data['end_time'] = datetime.fromisoformat(session_data['end_time'])
                        
                        # Create SessionStats object
                        session_stats = SessionStats(
                            session_id=session_data['session_id'],
                            start_time=session_data['start_time'],
                            end_time=session_data['end_time'],
                            total_trades=session_data['total_trades'],
                            winning_trades=session_data['winning_trades'],
                            losing_trades=session_data['losing_trades'],
                            win_rate=session_data['win_rate'],
                            avg_profit=session_data['avg_profit'],
                            max_drawdown=session_data['max_drawdown'],
                            sharpe_ratio=session_data['sharpe_ratio'],
                            psychological_state=session_data['psychological_state'],
                            technical_state=session_data['technical_state'],
                            reinforcement_stats=session_data.get('reinforcement_stats'),
                            prediction_accuracy=session_data.get('prediction_accuracy')
                        )
                        
                        loaded_sessions.append(session_stats)
                        
                except Exception as e:
                    self.logger.error(f"Error loading session file {session_file}: {str(e)}")
                    continue
            
            # Add loaded sessions to session manager
            for session in loaded_sessions:
                self.session_manager.save_session_stats(session)
                
            self.logger.info(f"Successfully loaded {len(loaded_sessions)} sessions")
            
            # Maintain session limits
            if len(loaded_sessions) > self.max_sessions:
                self.session_manager.maintain_min_sessions(self.min_sessions)
                
        except Exception as e:
            self.logger.error(f"Error loading existing sessions: {str(e)}")
            raise

    def save_session(self, session_stats: SessionStats):
        """Save session data to disk"""
        try:
            # Create filename with session ID and timestamp
            filename = f"session_{session_stats.session_id}_{session_stats.end_time.strftime('%Y%m%d_%H%M%S')}.json"
            filepath = os.path.join(self.session_save_dir, filename)
            
            # Convert session stats to dictionary
            session_dict = {
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
                'technical_state': session_stats.technical_state,
                'reinforcement_stats': session_stats.reinforcement_stats,
                'prediction_accuracy': session_stats.prediction_accuracy
            }
            
            # Save to file
            with open(filepath, 'w') as f:
                import json
                json.dump(session_dict, f, indent=4)
                
            self.logger.info(f"Session saved successfully: {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error saving session: {str(e)}")
            raise

    def _setup_session_logging(self, session_id: str) -> logging.Logger:
        """Setup session-specific logging"""
        try:
            # Create session log directory
            session_log_dir = os.path.join(self.log_dir, 'sessions', session_id)
            os.makedirs(session_log_dir, exist_ok=True)
            
            # Create session-specific log file
            log_file = os.path.join(session_log_dir, 'session.log')
            
            # Create session logger
            session_logger = logging.getLogger(f'session_{session_id}')
            session_logger.setLevel(logging.INFO)
            
            # Create handlers
            file_handler = logging.FileHandler(log_file)
            console_handler = logging.StreamHandler()
            
            # Create formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            
            # Set formatter for handlers
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            
            # Add handlers to logger
            session_logger.addHandler(file_handler)
            session_logger.addHandler(console_handler)
            
            return session_logger
            
        except Exception as e:
            self.logger.error(f"Error setting up session logging: {str(e)}")
            raise

    def _save_session_report(self, session: SessionStats):  # Removed logger parameter
        """Generate and save final session report"""
        try:
            # Get logger for this session
            logger = logging.getLogger(f'session_{session.session_id}')
            
            # Create report directory
            report_dir = os.path.join(self.log_dir, 'sessions', session.session_id, 'reports')
            os.makedirs(report_dir, exist_ok=True)
            
            # Generate report content
            report = {
                'session_summary': {
                    'id': session.session_id,
                    'start_time': session.start_time.isoformat(),
                    'end_time': session.end_time.isoformat(),
                    'duration': str(session.end_time - session.start_time)
                },
                'performance_metrics': {
                    'total_trades': session.total_trades,
                    'winning_trades': session.winning_trades,
                    'losing_trades': session.losing_trades,
                    'win_rate': f"{session.win_rate:.2%}",
                    'avg_profit': f"{session.avg_profit:.2%}",
                    'max_drawdown': f"{session.max_drawdown:.2%}",
                    'sharpe_ratio': f"{session.sharpe_ratio:.2f}"
                },
                'reinforcement_stats': session.reinforcement_stats,
                'psychological_state': session.psychological_state,
                'technical_state': session.technical_state
            }
            
            # Save JSON report
            report_file = os.path.join(report_dir, 'final_report.json')
            with open(report_file, 'w') as f:
                import json
                json.dump(report, f, indent=4)
                
            # Save human-readable summary
            summary_file = os.path.join(report_dir, 'summary.txt')
            with open(summary_file, 'w') as f:
                f.write("Trading Session Summary\n")
                f.write("=====================\n\n")
                f.write(f"Session ID: {session.session_id}\n")
                f.write(f"Duration: {session.end_time - session.start_time}\n\n")
                f.write("Performance Metrics\n")
                f.write("-----------------\n")
                f.write(f"Total Trades: {session.total_trades}\n")
                f.write(f"Win Rate: {session.win_rate:.2%}\n")
                f.write(f"Average Profit: {session.avg_profit:.2%}\n")
                f.write(f"Max Drawdown: {session.max_drawdown:.2%}\n")
                f.write(f"Sharpe Ratio: {session.sharpe_ratio:.2f}\n")
            
            logger.info(f"Session report saved to {report_dir}")
            
        except Exception as e:
            self.logger.error(f"Error saving session report: {str(e)}")
            raise

    def _generate_forecast(self, 
                          model: Model, 
                          state: np.ndarray, 
                          current_data: pd.Series) -> Dict:
        """
        Generate forecast from current state including price predictions,
        confidence metrics, and technical analysis.
        
        Args:
            model: The trained model to use for predictions
            state: Current market state as numpy array
            current_data: Current market data as pandas Series
        
        Returns:
            Dict containing predictions, confidence scores, and analysis
        """
        try:
            # Validate inputs
            if model is None:
                raise ValueError("Model cannot be None")
            if state is None or len(state) == 0:
                raise ValueError("Invalid state array")
            if current_data is None or 'close' not in current_data:
                raise ValueError("Invalid current data")

            # Ensure state is properly shaped and handle NaN values
            state_cleaned = np.nan_to_num(state.reshape(1, -1), nan=0.0)
            
            # Get Q-values for all actions
            try:
                q_values = model.predict_on_batch(state_cleaned)[0]
            except Exception as e:
                self.logger.error(f"Error getting Q-values from model: {str(e)}")
                q_values = np.zeros(3)  # Default to 3 actions: HOLD, BUY, SELL

            # Define action mapping with expected price movements
            action_mapping = {
                0: {'name': 'HOLD', 'movement': 0.0},
                1: {'name': 'BUY', 'movement': 0.01},  # 1% up movement
                2: {'name': 'SELL', 'movement': -0.01}  # 1% down movement
            }

            # Get best action and its index
            best_action_idx = int(np.argmax(q_values))
            predicted_movement = action_mapping[best_action_idx]['movement']

            # Get current price and ensure it's valid
            try:
                current_price = float(current_data['close'])
                if not np.isfinite(current_price):
                    raise ValueError("Invalid current price")
            except Exception as e:
                self.logger.error(f"Error getting current price: {str(e)}")
                current_price = 0.0

            # Generate multiple price predictions with increasing uncertainty
            predictions = []
            volatility = self._calculate_volatility(current_data)
            
            for i in range(1, 4):  # Generate 3 future predictions
                # Increase movement impact and uncertainty with time
                time_factor = i * 1.5
                movement_with_uncertainty = predicted_movement * time_factor
                
                # Add volatility-based noise
                noise = np.random.normal(0, volatility * i * 0.1)
                
                # Calculate predicted price
                predicted_price = current_price * (1 + movement_with_uncertainty + noise)
                predictions.append(float(np.clip(predicted_price, current_price * 0.9, current_price * 1.1)))

            # Calculate confidence metrics
            q_max = float(np.max(q_values))
            q_mean = float(np.mean(q_values))
            q_std = float(np.std(q_values))
            
            # Confidence based on Q-value separation and consistency
            q_confidence = float(np.clip((q_max - q_mean) / (q_std if q_std > 0 else 1), 0, 1))
            
            # Get technical analysis state
            technical_state = self._calculate_technical_state(
                pd.DataFrame([current_data])
            )
            
            # Adjust confidence based on technical indicators
            technical_confidence = self._calculate_technical_confidence(technical_state)
            
            # Combined confidence score
            confidence = float(np.clip((q_confidence + technical_confidence) / 2, 0, 1))

            # Create forecast dictionary
            forecast = {
                'values': predictions,
                'confidence': confidence,
                'technical_state': technical_state,
                'best_action': best_action_idx,
                'action_name': action_mapping[best_action_idx]['name'],
                'q_values': q_values.tolist(),
                'metadata': {
                    'q_confidence': q_confidence,
                    'technical_confidence': technical_confidence,
                    'volatility': volatility,
                    'timestamp': datetime.now().isoformat()
                }
            }

            # Log forecast generation
            self.logger.debug(
                f"Generated forecast - Action: {forecast['action_name']}, "
                f"Confidence: {confidence:.2f}, "
                f"Predictions: {predictions}"
            )

            return forecast

        except Exception as e:
            self.logger.error(f"Error generating forecast: {str(e)}")
            # Return safe default forecast
            return {
                'values': [float(current_data.get('close', 0))] * 3,
                'confidence': 0.0,
                'technical_state': {},
                'best_action': 0,  # HOLD
                'action_name': 'HOLD',
                'q_values': [0.0, 0.0, 0.0],
                'metadata': {
                    'q_confidence': 0.0,
                    'technical_confidence': 0.0,
                    'volatility': 0.0,
                    'timestamp': datetime.now().isoformat(),
                    'error': str(e)
                }
            }

    def _calculate_volatility(self, data: pd.Series) -> float:
        """Calculate price volatility from recent data"""
        try:
            if 'close' not in data:
                return 0.0
            
            # If we have historical data in the series
            if isinstance(data['close'], (pd.Series, np.ndarray)):
                returns = np.diff(data['close']) / data['close'][:-1]
                return float(np.std(returns))
            
            return 0.0
        except Exception as e:
            self.logger.error(f"Error calculating volatility: {str(e)}")
            return 0.0

    def _calculate_technical_confidence(self, technical_state: Dict) -> float:
        """Calculate confidence score based on technical indicators"""
        try:
            confidence_scores = []
            
            # Trend strength confidence
            if 'trend' in technical_state:
                trend_strength = abs(technical_state['trend'].get('strength', 0))
                confidence_scores.append(trend_strength)
            
            # Momentum confidence
            if 'momentum' in technical_state:
                momentum = technical_state['momentum']
                # RSI confidence (higher near extremes)
                rsi = momentum.get('rsi', 50)
                rsi_confidence = abs(rsi - 50) / 50
                confidence_scores.append(rsi_confidence)
                
                # MACD confidence
                macd = abs(momentum.get('macd', 0))
                macd_confidence = min(macd / 2, 1.0)  # Normalize MACD
                confidence_scores.append(macd_confidence)
            
            # Calculate final confidence
            if confidence_scores:
                return float(np.mean(confidence_scores))
            return 0.5  # Default moderate confidence
            
        except Exception as e:
            self.logger.error(f"Error calculating technical confidence: {str(e)}")
            return 0.5

    def _calculate_session_metrics(self,
                                session_id: str,
                                trades: List[Dict],
                                forecasts: List[Dict],
                                total_reward: float,
                                initial_balance: float) -> SessionStats:
        """
        Calculate comprehensive session metrics with safety checks.
        
        Args:
            session_id: Unique identifier for the session
            trades: List of trade dictionaries containing profit/loss info
            forecasts: List of forecast dictionaries with predictions
            total_reward: Total reinforcement learning reward for session
            initial_balance: Starting balance for the session
        
        Returns:
            SessionStats object containing all calculated metrics
        """
        try:
            # Initialize timing
            start_time = datetime.now() - timedelta(hours=1)  # Assume 1 hour session
            end_time = datetime.now()
            
            # Trade Performance Metrics
            total_trades = len(trades)
            winning_trades = sum(1 for t in trades if t.get('trade_profit', 0) > 0)
            losing_trades = total_trades - winning_trades
            
            # Win Rate Calculation
            win_rate = winning_trades / max(total_trades, 1)  # Avoid division by zero
            
            # Profit Metrics
            profits = [t.get('trade_profit', 0) for t in trades]
            avg_profit = float(np.mean(profits)) if profits else 0.0
            total_profit = float(np.sum(profits))
            
            # Risk Metrics
            max_profit = float(max(profits)) if profits else 0.0
            max_loss = float(min(profits)) if profits else 0.0
            profit_factor = (
                abs(sum(p for p in profits if p > 0)) / 
                abs(sum(p for p in profits if p < 0))
                if any(p < 0 for p in profits) else float('inf')
            )

            # Calculate Running Balance and Drawdown
            balance_curve = [initial_balance]
            peak = initial_balance
            drawdowns = []
            
            for trade in trades:
                profit = trade.get('trade_profit', 0)
                current_balance = balance_curve[-1] * (1 + profit)
                balance_curve.append(current_balance)
                
                peak = max(peak, current_balance)
                drawdown = (peak - current_balance) / peak
                drawdowns.append(drawdown)
            
            # Maximum Drawdown
            max_drawdown = float(max(drawdowns)) if drawdowns else 0.0
            
            # Calculate Returns
            returns = np.diff(balance_curve) / balance_curve[:-1]
            
            # Risk-Adjusted Metrics
            try:
                sharpe_ratio = float(
                    np.mean(returns) / np.std(returns) * np.sqrt(252)  # Annualized
                    if len(returns) > 1 and np.std(returns) > 0 else 0.0
                )
                
                sortino_ratio = float(
                    np.mean(returns) / np.std([r for r in returns if r < 0]) * np.sqrt(252)
                    if any(r < 0 for r in returns) else float('inf')
                )
            except Exception as e:
                self.logger.warning(f"Error calculating risk metrics: {str(e)}")
                sharpe_ratio = 0.0
                sortino_ratio = 0.0
            
            # Forecast Performance
            if forecasts:
                forecast_accuracy = np.mean([
                    f['confidence'] for f in forecasts
                    if isinstance(f.get('confidence'), (int, float))
                ])
                
                directional_accuracy = np.mean([
                    1 if (f.get('best_action') == 1 and t.get('trade_profit', 0) > 0) or
                        (f.get('best_action') == 2 and t.get('trade_profit', 0) < 0) else 0
                    for f, t in zip(forecasts, trades)
                    if f.get('best_action') in [1, 2]  # Only count BUY/SELL predictions
                ]) if trades and forecasts else 0.0
            else:
                forecast_accuracy = 0.0
                directional_accuracy = 0.0
            
            # Get final psychological and technical states
            final_psych_state = self._get_final_psychological_state(trades, forecasts)
            final_tech_state = self._get_final_technical_state(trades, forecasts)
            
            # Calculate reinforcement learning metrics
            rl_metrics = {
                'total_reward': float(total_reward),
                'avg_reward': float(total_reward / max(len(forecasts), 1)),
                'final_epsilon': float(self.epsilon),
                'forecast_accuracy': float(forecast_accuracy),
                'directional_accuracy': float(directional_accuracy)
            }
            
            # Create comprehensive session statistics
            session_stats = SessionStats(
                session_id=session_id,
                start_time=start_time,
                end_time=end_time,
                total_trades=total_trades,
                winning_trades=winning_trades,
                losing_trades=losing_trades,
                win_rate=win_rate,
                avg_profit=avg_profit,
                max_drawdown=max_drawdown,
                sharpe_ratio=sharpe_ratio,
                psychological_state=final_psych_state,
                technical_state=final_tech_state,
                reinforcement_stats=rl_metrics,
                prediction_accuracy=forecast_accuracy,
                model=None,  # Model can be attached later if needed
            )
            
            # Additional metrics for logging
            extended_metrics = {
                'total_profit': total_profit,
                'profit_factor': profit_factor,
                'max_profit': max_profit,
                'max_loss': max_loss,
                'sortino_ratio': sortino_ratio,
                'final_balance': balance_curve[-1],
                'return_pct': (balance_curve[-1] - initial_balance) / initial_balance * 100,
                'avg_trade_duration': np.mean([
                    t.get('duration', 0) for t in trades
                    if isinstance(t.get('duration'), (int, float))
                ]) if trades else 0
            }
            
            # Log detailed metrics
            self.logger.info(
                f"Session {session_id} metrics calculated:\n"
                f"Win Rate: {win_rate:.2%}\n"
                f"Total Profit: {total_profit:.2f}\n"
                f"Max Drawdown: {max_drawdown:.2%}\n"
                f"Sharpe Ratio: {sharpe_ratio:.2f}\n"
                f"Forecast Accuracy: {forecast_accuracy:.2%}"
            )
            
            # Store extended metrics in session manager
            self.session_manager.store_extended_metrics(session_id, extended_metrics)
            
            return session_stats
            
        except Exception as e:
            self.logger.error(f"Error calculating session metrics: {str(e)}")
            # Return safe default metrics
            return SessionStats(
                session_id=session_id,
                start_time=datetime.now() - timedelta(hours=1),
                end_time=datetime.now(),
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0.0,
                avg_profit=0.0,
                max_drawdown=0.0,
                sharpe_ratio=0.0,
                psychological_state=self._get_initial_psychological_state(),
                technical_state=self._get_initial_technical_state(),
                reinforcement_stats={'total_reward': 0.0, 'avg_reward': 0.0},
                prediction_accuracy=0.0
            )

    def _get_final_psychological_state(self, trades: List[Dict], forecasts: List[Dict]) -> Dict:
        """Calculate final psychological state based on trading performance"""
        try:
            # Base confidence on recent performance
            recent_trades = trades[-10:] if len(trades) > 10 else trades
            recent_win_rate = np.mean([
                1 if t.get('trade_profit', 0) > 0 else 0 
                for t in recent_trades
            ]) if recent_trades else 0.5
            
            # Calculate stress level based on drawdowns
            recent_drawdowns = [t.get('drawdown', 0) for t in recent_trades]
            stress_level = float(np.mean(recent_drawdowns)) if recent_drawdowns else 0.3
            
            # Calculate risk tolerance based on position sizes
            position_sizes = [t.get('position_size', 0) for t in recent_trades]
            risk_tolerance = float(np.mean(position_sizes)) if position_sizes else 0.5
            
            return {
                'confidence': float(recent_win_rate),
                'emotional_balance': float(max(0, 1 - stress_level)),
                'risk_tolerance': float(risk_tolerance),
                'stress_level': float(stress_level)
            }
        except Exception as e:
            self.logger.error(f"Error calculating psychological state: {str(e)}")
            return self._get_initial_psychological_state()

    def _get_final_technical_state(self, trades: List[Dict], forecasts: List[Dict]) -> Dict:
        """Calculate final technical state based on recent market conditions"""
        try:
            # Use most recent forecast's technical state if available
            if forecasts and 'technical_state' in forecasts[-1]:
                return forecasts[-1]['technical_state']
            
            # Fallback to calculating from trades
            recent_prices = [t.get('price', 0) for t in trades[-20:]]
            if len(recent_prices) > 1:
                trend_direction = 'up' if recent_prices[-1] > recent_prices[0] else 'down'
                trend_strength = abs(recent_prices[-1] - recent_prices[0]) / recent_prices[0]
                
                return {
                    'trend': {
                        'direction': trend_direction,
                        'strength': float(trend_strength)
                    },
                    'momentum': float(trend_strength),
                    'volatility': float(np.std(recent_prices) / np.mean(recent_prices))
                }
                
            return self._get_initial_technical_state()
            
        except Exception as e:
            self.logger.error(f"Error calculating technical state: {str(e)}")
            return self._get_initial_technical_state()








