from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
from datetime import datetime
import numpy as np
import json
import os
import shutil
from pathlib import Path

@dataclass
class ForecastLog:
    """Tracks individual forecast details"""
    timestamp: datetime
    predicted_values: List[float]
    confidence: float
    technical_indicators: Dict
    actual_value: Optional[float] = None
    forecast_error: Optional[float] = None

@dataclass
class ReinforcementUpdate:
    """Tracks reinforcement learning updates"""
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    q_values_before: np.ndarray
    q_values_after: np.ndarray
    epsilon: float
    loss: float

@dataclass
class SessionLog:
    """Tracks individual session details"""
    session_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    initial_balance: float = 100000.0
    final_balance: Optional[float] = None
    trades: List[Dict] = field(default_factory=list)
    forecasts: List[ForecastLog] = field(default_factory=list)
    reinforcement_updates: List[ReinforcementUpdate] = field(default_factory=list)
    performance_metrics: Dict = field(default_factory=dict)
    psychological_state: Dict = field(default_factory=dict)
    technical_state: Dict = field(default_factory=dict)

@dataclass
class EpisodeLog:
    """Tracks episode-level details"""
    episode_id: int
    training_session_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    sessions: List[SessionLog] = field(default_factory=list)
    episode_metrics: Dict = field(default_factory=dict)
    cumulative_reward: float = 0.0
    average_q_value: float = 0.0
    epsilon: float = 0.0
    performance_score: float = 0.0  # Combined performance metric

@dataclass
class TrainingSessionLog:
    """Tracks complete training session"""
    session_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    episodes: List[EpisodeLog] = field(default_factory=list)
    best_episode_id: Optional[int] = None
    configuration: Dict = field(default_factory=dict)
    session_metrics: Dict = field(default_factory=dict)

class EnhancedTrainingLogger:
    """Logger for managing training sessions and episodes logs"""
    
    def __init__(self, base_log_dir: str):
        self.base_log_dir = Path(base_log_dir)
        self.current_training_session: Optional[TrainingSessionLog] = None
        self.current_episode: Optional[EpisodeLog] = None
        self.current_session: Optional[SessionLog] = None
        
        # Create directory structure
        self._create_directory_structure()
        
    def _create_directory_structure(self):
        """Create organized directory structure for logs"""
        directories = [
            self.base_log_dir,
            self.base_log_dir / 'training_sessions',
            self.base_log_dir / 'episodes',
            self.base_log_dir / 'sessions'
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            
    def start_training_session(self, configuration: Dict) -> str:
        """Start a new training session"""
        session_id = f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.current_training_session = TrainingSessionLog(
            session_id=session_id,
            start_time=datetime.now(),
            configuration=configuration
        )
        
        # Create session directory
        (self.base_log_dir / 'training_sessions' / session_id).mkdir(parents=True, exist_ok=True)
        
        return session_id
        
    def start_episode(self, episode_id: int) -> EpisodeLog:
        """Start a new episode within current training session"""
        if self.current_training_session is None:
            raise ValueError("No active training session")
            
        self.current_episode = EpisodeLog(
            episode_id=episode_id,
            training_session_id=self.current_training_session.session_id,
            start_time=datetime.now()
        )
        
        return self.current_episode
        
    def start_session(self, session_id: str, initial_balance: float) -> SessionLog:
        """Start a new trading session within current episode"""
        if self.current_episode is None:
            raise ValueError("No active episode")
            
        self.current_session = SessionLog(
            session_id=session_id,
            start_time=datetime.now(),
            initial_balance=initial_balance
        )
        
        return self.current_session
        
    def log_reinforcement_update(self, state: np.ndarray, action: int, reward: float,
                               next_state: np.ndarray, q_values_before: np.ndarray,
                               q_values_after: np.ndarray, epsilon: float, loss: float):
        """Log reinforcement learning update"""
        if self.current_session is None:
            raise ValueError("No active session")
            
        update = ReinforcementUpdate(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            q_values_before=q_values_before,
            q_values_after=q_values_after,
            epsilon=epsilon,
            loss=loss
        )
        
        self.current_session.reinforcement_updates.append(update)
        
    def log_forecast(self, predicted_values: List[float], confidence: float,
                    technical_indicators: Dict, actual_value: Optional[float] = None):
        """Log a forecast"""
        if self.current_session is None:
            raise ValueError("No active session")
            
        forecast = ForecastLog(
            timestamp=datetime.now(),
            predicted_values=predicted_values,
            confidence=confidence,
            technical_indicators=technical_indicators,
            actual_value=actual_value
        )
        
        self.current_session.forecasts.append(forecast)
        
    def end_session(self, final_balance: float, performance_metrics: Dict,
                   psychological_state: Dict, technical_state: Dict):
        """End current session and save metrics"""
        if self.current_session is None:
            raise ValueError("No active session")
            
        self.current_session.end_time = datetime.now()
        self.current_session.final_balance = final_balance
        self.current_session.performance_metrics = performance_metrics
        self.current_session.psychological_state = psychological_state
        self.current_session.technical_state = technical_state
        
        # Save session log
        self._save_session_log(self.current_session)
        self.current_session = None
        
    def end_episode(self):
        """End current episode"""
        if self.current_episode is None:
            raise ValueError("No active episode")
            
        self.current_episode.end_time = datetime.now()
        
        # Save episode log
        self._save_episode_log(self.current_episode)
        self.current_episode = None
        
    def end_training_session(self):
        """End current training session"""
        if self.current_training_session is None:
            raise ValueError("No active training session")
            
        self.current_training_session.end_time = datetime.now()
        
        # Save training session log
        self._save_training_session_log(self.current_training_session)
        self.current_training_session = None

    def _save_session_log(self, session: SessionLog):
        """Save session log to file"""
        session_dir = self.base_log_dir / 'sessions' / session.session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        
        with open(session_dir / 'session.json', 'w') as f:
            json.dump(self._convert_session_to_dict(session), f, indent=2)
            
    def _save_episode_log(self, episode: EpisodeLog):
        """Save episode log to file"""
        episode_dir = self.base_log_dir / 'episodes' / f"{episode.training_session_id}_ep{episode.episode_id}"
        episode_dir.mkdir(parents=True, exist_ok=True)
        
        with open(episode_dir / 'episode.json', 'w') as f:
            json.dump(self._convert_episode_to_dict(episode), f, indent=2)
            
    def _save_training_session_log(self, training_session: TrainingSessionLog):
        """Save training session log to file"""
        session_dir = self.base_log_dir / 'training_sessions' / training_session.session_id
        
        with open(session_dir / 'session_summary.json', 'w') as f:
            json.dump(self._convert_training_session_to_dict(training_session), f, indent=2)

    def _convert_session_to_dict(self, session: SessionLog) -> Dict:
        """Convert session log to dictionary for saving"""
        return {
            'session_id': session.session_id,
            'start_time': session.start_time.isoformat(),
            'end_time': session.end_time.isoformat() if session.end_time else None,
            'initial_balance': session.initial_balance,
            'final_balance': session.final_balance,
            'performance_metrics': session.performance_metrics,
            'psychological_state': session.psychological_state,
            'technical_state': session.technical_state
        }

    def _convert_episode_to_dict(self, episode: EpisodeLog) -> Dict:
        """Convert episode log to dictionary for saving"""
        return {
            'episode_id': episode.episode_id,
            'training_session_id': episode.training_session_id,
            'start_time': episode.start_time.isoformat(),
            'end_time': episode.end_time.isoformat() if episode.end_time else None,
            'sessions': [self._convert_session_to_dict(s) for s in episode.sessions]
        }

    def _convert_training_session_to_dict(self, training_session: TrainingSessionLog) -> Dict:
        """Convert training session log to dictionary for saving"""
        return {
            'session_id': training_session.session_id,
            'start_time': training_session.start_time.isoformat(),
            'end_time': training_session.end_time.isoformat() if training_session.end_time else None,
            'configuration': training_session.configuration,
            'episodes': [self._convert_episode_to_dict(e) for e in training_session.episodes]
        }
