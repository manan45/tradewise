import os
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from dataclasses import dataclass

from app.core.ai.technical_analyzer import TechnicalPatternAnalyzer
from app.core.ai.market_environment import MarketEnvironment
from app.core.ai.trading_session import TradingSession
from app.core.ai.zone_analyzer import ZonePatternAnalyzer
from app.core.ai.market_psychology import PsychologyPatternAnalyzer
from app.core.ai.trading_psychology import TradingPsychology
from app.core.ai.model_builder import ModelBuilder
from app.core.ai.session_manager import SessionManager
from app.core.ai.reinforcement.market_state import MarketState
from app.core.ai.reinforcement.action_space import ActionSpace
from app.core.ai.reinforcement.reward_calculator import RewardCalculator
from app.core.domain.models.trade_suggestion import DetailedTradeSuggestion

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
        self.session_manager = SessionManager(max_sessions=max_sessions)
        self.model_builder = ModelBuilder()
        
        # Initialize market environment components
        self.market_state = MarketState()
        self.action_space = ActionSpace()
        self.reward_calculator = RewardCalculator()
        
        # RL parameters
        self.gamma = 0.95  # Discount factor
        self.epsilon = 0.1  # Exploration rate
        self.learning_rate = 0.001
        
        # Setup logging and directories
        self._setup_logging()
        self._create_directories()
        
        # Load existing sessions if available
        self._load_existing_sessions()

    async def train(self, data_feed: pd.DataFrame) -> SessionStats:
        """Train model with market environment-based reinforcement learning"""
        try:
            self.logger.info("Starting new training session with market environment")
            
            # Initialize market environment
            market_env = self._initialize_market_environment(data_feed)
            
            # Create new session
            session_id = str(datetime.now().timestamp())
            session_stats = SessionStats(
                session_id=session_id,
                start_time=datetime.now(),
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
                model=None,
                reinforcement_stats={
                    'total_rewards': 0,
                    'avg_reward': 0,
                    'max_reward': float('-inf'),
                    'min_reward': float('inf'),
                    'episode_rewards': []
                }
            )

            # Train with reinforcement learning
            trained_session = await self._train_with_market_environment(
                session_stats,
                market_env,
                data_feed
            )
            
            # Update session manager
            self.session_manager.save_session_stats(trained_session)
            
            return trained_session
            
        except Exception as e:
            self.logger.error(f"Error during market environment training: {str(e)}")
            raise

    async def _train_with_market_environment(self,
                                          session: SessionStats,
                                          market_env: MarketEnvironment,
                                          data_feed: pd.DataFrame) -> SessionStats:
        """Train model using market environment reinforcement learning"""
        try:
            # Initialize model
            model = self._build_model(market_env.state_size, market_env.action_size)
            session.model = model
            
            # Training parameters
            episodes = 100
            max_steps = len(data_feed) - 1
            
            for episode in range(episodes):
                self.logger.info(f"Starting episode {episode + 1}/{episodes}")
                
                # Reset environment
                state = market_env.reset()
                total_reward = 0
                
                for step in range(max_steps):
                    # Get action using epsilon-greedy policy
                    if np.random.random() < self.epsilon:
                        action = np.random.randint(market_env.action_size)
                    else:
                        action = np.argmax(model.predict(state.reshape(1, -1))[0])
                    
                    # Take action in environment
                    next_state, reward, done, info = market_env.step(action)
                    
                    # Store experience and train model
                    self._train_on_experience(
                        model,
                        state,
                        action,
                        reward,
                        next_state,
                        done
                    )
                    
                    # Update state and accumulate reward
                    state = next_state
                    total_reward += reward
                    
                    # Update session stats
                    self._update_session_stats(session, reward, info)
                    
                    if done:
                        break
                
                # Update reinforcement stats
                session.reinforcement_stats['episode_rewards'].append(total_reward)
                session.reinforcement_stats['total_rewards'] += total_reward
                session.reinforcement_stats['avg_reward'] = (
                    session.reinforcement_stats['total_rewards'] / (episode + 1)
                )
                session.reinforcement_stats['max_reward'] = max(
                    session.reinforcement_stats['max_reward'],
                    total_reward
                )
                session.reinforcement_stats['min_reward'] = min(
                    session.reinforcement_stats['min_reward'],
                    total_reward
                )
                
                # Decay epsilon
                self.epsilon = max(0.01, self.epsilon * 0.995)
            
            return session
            
        except Exception as e:
            self.logger.error(f"Error during market environment training: {str(e)}")
            raise

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
                           model,
                           state: np.ndarray,
                           action: int,
                           reward: float,
                           next_state: np.ndarray,
                           done: bool):
        """Train model on single experience tuple"""
        try:
            # Get current Q-values
            current_q = model.predict(state.reshape(1, -1))[0]
            
            # Get next Q-values
            next_q = model.predict(next_state.reshape(1, -1))[0]
            
            # Update Q-value for taken action
            if done:
                current_q[action] = reward
            else:
                current_q[action] = reward + self.gamma * np.max(next_q)
            
            # Train model
            model.fit(
                state.reshape(1, -1),
                current_q.reshape(1, -1),
                verbose=0
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
            # Update trade counts
            if info.get('trade_executed', False):
                session.total_trades += 1
                if info.get('trade_profit', 0) > 0:
                    session.winning_trades += 1
                else:
                    session.losing_trades += 1
            
            # Update win rate
            if session.total_trades > 0:
                session.win_rate = session.winning_trades / session.total_trades
            
            # Update profit metrics
            current_profit = info.get('trade_profit', 0)
            session.avg_profit = (
                (session.avg_profit * (session.total_trades - 1) + current_profit)
                / session.total_trades if session.total_trades > 0 else 0
            )
            
            # Update drawdown
            if info.get('drawdown', 0) > session.max_drawdown:
                session.max_drawdown = info['drawdown']
            
            # Update Sharpe ratio
            if len(session.reinforcement_stats['episode_rewards']) > 1:
                returns = np.array(session.reinforcement_stats['episode_rewards'])
                session.sharpe_ratio = (
                    np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
                )
            
        except Exception as e:
            self.logger.error(f"Error updating session stats: {str(e)}")
            raise

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

