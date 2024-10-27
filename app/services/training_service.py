import os
import asyncio
import logging
from typing import Dict, List, Optional
import pandas as pd
from datetime import datetime
import sys
import argparse
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from app.core.ai.tradewise_ai import TradewiseAI
from app.core.domain.models.session_models import SessionStats
from app.core.domain.models.trade_suggestion import DetailedTradeSuggestion
from app.core.ai.model_builder import ModelBuilder, TrainingResult
from app.core.repositories.stock_repository import StockRepository
from app.core.domain.entities.stock import StockPrice

class TrainingService:
    """Service for training and evaluating TradewiseAI models"""
    
    def __init__(self, 
                 model_path: str = "./models/",
                 session_save_dir: str = "./sessions/",
                 log_dir: str = "./logs/"):
        self.tradewise = TradewiseAI(
            model_path=model_path,
            session_save_dir=session_save_dir,
            log_dir=log_dir
        )
        self.model_builder = ModelBuilder(
            sequence_length=10,
            prediction_horizon=1,
            batch_size=32,
            epochs=10
        )
        self.logger = logging.getLogger(__name__)

    def _convert_stockprice_to_dataframe(self, price_history: List[StockPrice]) -> pd.DataFrame:
        """Convert StockPrice list to DataFrame with technical indicators"""
        try:
            # Convert to DataFrame
            df = pd.DataFrame([{
                'timestamp': price.timestamp,
                'open': float(price.open),
                'high': float(price.high),
                'low': float(price.low),
                'close': float(price.close),
                'volume': float(price.volume)
            } for price in price_history])
            
            # Sort by timestamp
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Calculate technical indicators
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['sma_50'] = df['close'].rolling(window=50).mean()
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            exp1 = df['close'].ewm(span=12, adjust=False).mean()
            exp2 = df['close'].ewm(span=26, adjust=False).mean()
            df['macd'] = exp1 - exp2
            df['macd_hist'] = df['macd'] - df['macd'].ewm(span=9, adjust=False).mean()
            
            # Bollinger Bands
            df['bb_middle'] = df['close'].rolling(window=20).mean()
            bb_std = df['close'].rolling(window=20).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
            df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # Volatility and Trend
            df['volatility'] = df['close'].rolling(window=20).std() / df['close'].rolling(window=20).mean()
            df['trend_strength'] = abs(df['sma_20'] - df['sma_50']) / df['sma_50']
            df['price_momentum'] = df['close'].pct_change(periods=10)
            
            # ATR
            high_low = df['high'] - df['low']
            high_close = abs(df['high'] - df['close'].shift())
            low_close = abs(df['low'] - df['close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            df['atr'] = true_range.rolling(window=14).mean()
            
            # Forward fill NaN values
            df = df.fillna(method='ffill').fillna(method='bfill')
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error converting price history to DataFrame: {str(e)}")
            raise
        
    async def train_and_evaluate(self, 
                               symbol: str,
                               price_history: List[StockPrice]) -> Dict:
        """Train model, evaluate performance, and generate trade suggestions"""
        try:
            # Convert price history to DataFrame with technical indicators
            data_feed = self._convert_stockprice_to_dataframe(price_history)
            
            if data_feed.empty:
                raise ValueError(f"No valid data available for {symbol}")
            
            # Train model and get best session
            best_session = await self.tradewise.train(data_feed)
            
            # Generate trade suggestions using best session
            suggestions = await self.tradewise.generate_trade_suggestions(data_feed)
            
            return {
                'status': 'success',
                'timestamp': datetime.now().isoformat(),
                'training_summary': {
                    'session_performance': {
                        'win_rate': best_session.win_rate,
                        'sharpe_ratio': best_session.sharpe_ratio,
                        'max_drawdown': best_session.max_drawdown
                    }
                },
                'trade_suggestions': [
                    suggestion.to_dict() for suggestion in suggestions
                ]
            }
            
        except Exception as e:
            self.logger.error(f"Error in training and evaluation: {str(e)}")
            raise

    async def _train_lstm_model(self, data_feed: pd.DataFrame) -> TrainingResult:
        """Train LSTM model asynchronously"""
        return await asyncio.to_thread(
            self.model_builder.train_model,
            data_feed
        )

    def _prepare_evaluation_results(self,
                                  training_result: TrainingResult,
                                  session_stats: SessionStats,
                                  suggestions: List[DetailedTradeSuggestion]) -> Dict:
        """Prepare evaluation results dictionary"""
        return {
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'training_summary': {
                'lstm_metrics': training_result.performance_metrics,
                'validation_metrics': training_result.validation_metrics,
                'session_performance': {
                    'win_rate': session_stats.win_rate,
                    'sharpe_ratio': session_stats.sharpe_ratio,
                    'max_drawdown': session_stats.max_drawdown
                }
            },
            'trade_suggestions': [
                suggestion.to_dict() for suggestion in suggestions
            ]
        }

async def train_and_evaluate_model(symbol: str) -> Dict:
    """Train and evaluate model for a specific symbol"""
    try:
        logger.info(f"Starting training for symbol: {symbol}")
        service = create_training_service()
        repository = StockRepository()
        
        # Get price history from database
        price_history = await repository.get_price_history(symbol)
        
        if not price_history:
            raise ValueError(f"No price history available for {symbol}")
        
        logger.info(f"Retrieved {len(price_history)} price points for {symbol}")
        return await service.train_and_evaluate(symbol, price_history)
        
    except Exception as e:
        logger.error(f"Error in train_and_evaluate_model: {str(e)}")
        return {
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

def create_training_service():
    """Create and configure training service with default paths"""
    # Ensure directories exist
    base_dir = Path(__file__).parent.parent.parent
    
    model_path = Path(os.getenv('MODEL_PATH', base_dir / 'models'))
    session_save_dir = Path(os.getenv('SESSION_SAVE_DIR', base_dir / 'sessions'))
    log_dir = Path(os.getenv('LOG_DIR', base_dir / 'logs'))
    
    # Create directories if they don't exist
    for directory in [model_path, session_save_dir, log_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    
    return TrainingService(
        model_path=str(model_path),
        session_save_dir=str(session_save_dir),
        log_dir=str(log_dir)
    )

def main():
    """Main entry point for training service"""
    try:
        # Set up argument parser
        parser = argparse.ArgumentParser(description='Train and evaluate TradewiseAI model')
        parser.add_argument('--symbol', type=str, default='AAPL',
                          help='Stock symbol to train on (default: AAPL)')
        args = parser.parse_args()
        
        logger.info(f"Starting training service for symbol: {args.symbol}")
        
        # Run training and evaluation
        results = asyncio.run(train_and_evaluate_model(args.symbol))
        
        # Print results
        print("\nTraining and Evaluation Results:")
        print("================================")
        print(f"Status: {results['status']}")
        print(f"Timestamp: {results['timestamp']}")
        
        if results['status'] == 'success':
            print("\nSession Performance:")
            print("-------------------")
            session_perf = results['training_summary']['session_performance']
            print(f"Win Rate: {session_perf['win_rate']:.2%}")
            print(f"Sharpe Ratio: {session_perf['sharpe_ratio']:.2f}")
            print(f"Max Drawdown: {session_perf['max_drawdown']:.2%}")
            
            print("\nTrade Suggestions:")
            print("----------------")
            for suggestion in results['trade_suggestions']:
                print(f"\nAction: {suggestion['action']}")
                print(f"Confidence: {suggestion['summary']['confidence']}")
                print(f"Risk Management: {suggestion['risk_management']}")
                print(f"Technical Analysis: {suggestion['technical_analysis']}")
        else:
            print(f"\nError: {results.get('error', 'Unknown error occurred')}")
            
    except KeyboardInterrupt:
        logger.info("Training service stopped by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error in main: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()
