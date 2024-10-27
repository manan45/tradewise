import os
import logging
from typing import Dict, List, Optional
from datetime import datetime
import json
import pandas as pd
import numpy as np
from dataclasses import dataclass

from .technical_indicators import TechnicalIndicatorCalculator
from .market_environment import MarketEnvironment
from .price_zone_analyzer import PriceZoneAnalyzer
from .market_psychology import MarketPsychology
from .trading_psychology import TradingPsychology
from .model_builder import build_lstm_model
from .psychology_analyzer import PsychologyPatternAnalyzer
from .technical_analyzer import TechnicalPatternAnalyzer
from .zone_analyzer import ZonePatternAnalyzer
from .session_manager import SessionManager

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
                 log_dir: str = "logs/"):
        """Initialize TradewiseAI"""
        self.model_path = model_path
        self.session_save_dir = session_save_dir
        self.log_dir = log_dir
        
        # Initialize components
        self.indicator_calculator = TechnicalIndicatorCalculator()
        self.price_analyzer = PriceZoneAnalyzer()
        self.trading_psychology = TradingPsychology()
        self.session_manager = SessionManager()
        
        # Setup logging
        self._setup_logging()
        
        # Create necessary directories
        os.makedirs(model_path, exist_ok=True)
        os.makedirs(session_save_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

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

    async def generate_trade_suggestions(self, symbol: str) -> List[Dict]:
        """Generate trade suggestions for a given symbol"""
        try:
            self.logger.info(f"Generating trade suggestions for {symbol}")
            
            # Get market data
            df = await self._get_market_data(symbol)
            if df.empty:
                raise ValueError(f"No data available for {symbol}")
            
            # Calculate indicators
            df_indicators = self.indicator_calculator.calculate_indicators(df)
            
            # Get market analysis
            analysis = self._analyze_market(df_indicators)
            
            # Generate predictions
            predictions = self._generate_predictions(df_indicators)
            
            # Format suggestions
            suggestions = self._format_suggestions(predictions, analysis)
            
            # Log session stats
            self._log_session_stats(symbol, suggestions)
            
            return suggestions
            
        except ValueError as ve:
            self.logger.error(f"ValueError: {str(ve)}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error: {str(e)}")
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
        """Analyze market conditions"""
        # Implementation details...
        pass

    def _generate_predictions(self, df: pd.DataFrame) -> List[Dict]:
        """Generate price predictions"""
        # Implementation details...
        pass

    def _format_suggestions(self, predictions: List[Dict], analysis: Dict) -> List[Dict]:
        """Format predictions into trade suggestions"""
        # Implementation details...
        pass

    def _log_session_stats(self, symbol: str, suggestions: List[Dict]):
        """Log session statistics"""
        # Implementation details...
        pass
