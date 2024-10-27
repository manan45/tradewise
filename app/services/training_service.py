import os
import asyncio
import logging
from typing import Dict, List, Optional
import pandas as pd
from datetime import datetime

from app.core.ai.tradewise_ai import TradewiseAI
from app.core.domain.models.trade_suggestion import DetailedTradeSuggestion
from app.core.ai.model_builder import ModelBuilder, TrainingResult
from app.core.repositories.stock_repository import StockRepository

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
            epochs=100,
            learning_rate=0.001
        )
        self.logger = logging.getLogger(__name__)
        
    async def train_and_evaluate(self, 
                               symbol: str,
                               data_feed: pd.DataFrame) -> Dict:
        """Train model, evaluate performance, and generate trade suggestions"""
        try:
            self.logger.info("Starting training and evaluation process")
            
            # Train LSTM model
            training_result = self.model_builder.train_model(data_feed)
            self.model_builder.save_model(training_result.model)
            
            # Train the model with reinforcement learning
            data_feed = data_feed.iloc[:-self.model_builder.prediction_horizon]
            sessions = await self.tradewise.generate_trade_suggestions(data_feed)
            
            # Generate trade suggestions
            suggestions = await self.tradewise.generate_trade_suggestions(data_feed)
            
            # Get performance metrics
            best_session = self.tradewise.session_manager.get_best_session()
            performance_summary = self.tradewise.session_manager.get_performance_summary()
            
            evaluation_results = {
                'status': 'success',
                'timestamp': datetime.now().isoformat(),
                'training_summary': {
                    'lstm_metrics': training_result.performance_metrics,
                    'validation_metrics': training_result.validation_metrics,
                    'total_sessions': len(self.tradewise.session_manager.sessions),
                    'best_session_performance': {
                        'session_id': best_session['session_id'],
                        'win_rate': best_session['win_rate'],
                        'sharpe_ratio': best_session['sharpe_ratio'],
                        'max_drawdown': best_session['max_drawdown']
                    } if best_session else None,
                    'overall_performance': performance_summary['performance_metrics']
                },
                'trade_suggestions': [
                    {
                        'suggestion': s.Suggestion,
                        'action': s.Action,
                        'summary': s.Summary,
                        'risk_management': s.Risk_Management,
                        'technical_analysis': s.Technical_Analysis,
                        'forecast_time': s.Forecast_Time
                    }
                    for s in suggestions
                ]
            }
            
            self.logger.info("Training and evaluation completed successfully")
            return evaluation_results
            
        except Exception as e:
            self.logger.error(f"Error in training and evaluation: {str(e)}")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    async def get_trade_suggestions(self, market_data: pd.DataFrame) -> List[DetailedTradeSuggestion]:
        """Get trade suggestions using trained model"""
        try:
            return await self.tradewise.generate_trade_suggestions(market_data)
        except Exception as e:
            self.logger.error(f"Error getting trade suggestions: {str(e)}")
            return []

def create_training_service():
    """Create and configure training service"""
    model_path = os.getenv('MODEL_PATH', './models/')
    session_save_dir = os.getenv('SESSION_SAVE_DIR', './sessions/')
    log_dir = os.getenv('LOG_DIR', './logs/')
    
    return TrainingService(
        model_path=model_path,
        session_save_dir=session_save_dir,
        log_dir=log_dir
    )

async def train_and_evaluate_model(symbol: str, 
                                 training_data: pd.DataFrame,
                                 evaluation_data: pd.DataFrame) -> Dict:
    """Train and evaluate model for a specific symbol"""
    service = create_training_service()
    repository = StockRepository()
    data_feed = await repository.get_price_history(symbol)
    return await service.train_and_evaluate(symbol, data_feed)

# CLI entry point
def main():
    """Main entry point for training service"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train and evaluate TradewiseAI model')
    parser.add_argument('--symbol', type=str, required=True, help='Stock symbol to train on')
    args = parser.parse_args()    
    # Run training and evaluation
    results = asyncio.run(
        train_and_evaluate_model(
            args.symbol
        )
    )
    
    # Print results
    print("\nTraining and Evaluation Results:")
    print("================================")
    print(f"Status: {results['status']}")
    print(f"Timestamp: {results['timestamp']}")
    
    if results['status'] == 'success':
        print("\nBest Session Performance:")
        print("------------------------")
        best_session = results['training_summary']['best_session_performance']
        if best_session:
            print(f"Session ID: {best_session['session_id']}")
            print(f"Win Rate: {best_session['win_rate']:.2%}")
            print(f"Sharpe Ratio: {best_session['sharpe_ratio']:.2f}")
            print(f"Max Drawdown: {best_session['max_drawdown']:.2%}")
        
        print("\nTrade Suggestions:")
        print("----------------")
        for suggestion in results['trade_suggestions']:
            print(f"\nAction: {suggestion['action']}")
            print(f"Confidence: {suggestion['summary']['confidence']}")
            print(f"Risk Management: {suggestion['risk_management']}")
            print(f"Technical Analysis: {suggestion['technical_analysis']}")

if __name__ == '__main__':
    main()
