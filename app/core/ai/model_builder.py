from typing import Dict, List, Optional, Tuple
import os
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from dataclasses import dataclass
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)

@dataclass
class TrainingResult:
    """Results from a training session"""
    model: Model
    performance_metrics: Dict[str, float]
    training_history: Dict[str, List[float]]
    validation_metrics: Dict[str, float]
    reinforcement_stats: Dict[str, float]

class ModelBuilder:
    """Handles LSTM model building, training and loading"""
    
    def __init__(self,
                 sequence_length: int = 60,
                 prediction_horizon: int = 24,
                 batch_size: int = 32,
                 epochs: int = 50):
        """Initialize model builder with training parameters"""
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.batch_size = batch_size
        self.epochs = epochs
        self.scaler = MinMaxScaler()
        
        # Reinforcement learning parameters
        self.reward_threshold = 0.7
        self.learning_rate = 0.001
        self.epsilon = 0.1  # Exploration rate

    def build_lstm_model(self, input_shape: Tuple[int, int]) -> Model:
        """Build LSTM model with specified input shape"""
        try:
            inputs = Input(shape=input_shape)
            x = LSTM(128, return_sequences=True)(inputs)
            x = Dropout(0.3)(x)
            x = LSTM(64, return_sequences=True)(x)
            x = Dropout(0.2)(x)
            x = LSTM(32, return_sequences=False)(x)
            x = Dropout(0.2)(x)
            x = Dense(16, activation='relu')(x)
            outputs = Dense(1)(x)
            
            model = Model(inputs=inputs, outputs=outputs)
            model.compile(
                optimizer=Adam(learning_rate=self.learning_rate),
                loss='huber',
                metrics=['mae', 'mse']
            )
            
            return model
            
        except Exception as e:
            logger.error(f"Error building LSTM model: {str(e)}")
            raise

    def build_rl_model(self, state_size: int, action_size: int) -> Model:
        """Build model for reinforcement learning"""
        try:
            model = Sequential([
                Dense(64, activation='relu', input_shape=(state_size,)),
                Dropout(0.2),
                Dense(32, activation='relu'),
                Dropout(0.2),
                Dense(action_size, activation='linear')
            ])
            
            model.compile(
                optimizer=Adam(learning_rate=self.learning_rate),
                loss='mse',
                metrics=['accuracy']
            )
            
            return model
            
        except Exception as e:
            logger.error(f"Error building RL model: {str(e)}")
            raise

    def prepare_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for training"""
        try:
            # Scale the data
            scaled_data = self.scaler.fit_transform(data[['close', 'volume']].values)
            
            X, y = [], []
            for i in range(len(scaled_data) - self.sequence_length - self.prediction_horizon):
                X.append(scaled_data[i:(i + self.sequence_length)])
                y.append(scaled_data[i + self.sequence_length:i + self.sequence_length + self.prediction_horizon, 0])
                
            return np.array(X), np.array(y)
            
        except Exception as e:
            logger.error(f"Error preparing data: {str(e)}")
            raise

    def train_model(self, data_feed: pd.DataFrame) -> TrainingResult:
        """Train the model with reinforcement learning"""
        try:
            evaluation_data = data_feed.iloc[-self.prediction_horizon:]
            train_data = data_feed.iloc[:-self.prediction_horizon]
            
            # Prepare data
            X_train, y_train = self.prepare_data(train_data)
            X_val, y_val = self.prepare_data(evaluation_data)
            
            # Build model
            model = self.build_lstm_model((self.sequence_length, X_train.shape[2]))
            
            # Initial training
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=self.epochs,
                batch_size=self.batch_size,
                verbose=0
            )
            
            # Calculate metrics
            performance_metrics = {
                'mse': history.history['loss'][-1],
                'mae': history.history['mae'][-1],
                'val_mse': history.history['val_loss'][-1],
                'val_mae': history.history['val_mae'][-1]
            }
            
            validation_metrics = self._calculate_validation_metrics(
                model, X_val, y_val, evaluation_data
            )
            
            return TrainingResult(
                model=model,
                performance_metrics=performance_metrics,
                training_history=history.history,
                validation_metrics=validation_metrics,
                reinforcement_stats={}
            )
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise

    def _calculate_validation_metrics(self, model: Model, X_val: np.ndarray,
                                   y_val: np.ndarray, evaluation_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate comprehensive validation metrics"""
        try:
            predictions = model.predict(X_val)
            
            unscaled_predictions = self.scaler.inverse_transform(
                np.column_stack((predictions, np.zeros_like(predictions)))
            )[:, 0]
            unscaled_actual = self.scaler.inverse_transform(
                np.column_stack((y_val, np.zeros_like(y_val)))
            )[:, 0]
            
            mse = np.mean((unscaled_predictions - unscaled_actual) ** 2)
            mae = np.mean(np.abs(unscaled_predictions - unscaled_actual))
            
            pred_directions = np.diff(unscaled_predictions) > 0
            actual_directions = np.diff(unscaled_actual) > 0
            directional_accuracy = np.mean(pred_directions == actual_directions)
            
            return {
                'mse': float(mse),
                'mae': float(mae),
                'directional_accuracy': float(directional_accuracy),
                'average_prediction_error': float(mae / np.mean(unscaled_actual))
            }
            
        except Exception as e:
            logger.error(f"Error calculating validation metrics: {str(e)}")
            return {}

    @staticmethod
    def load_model(model_path: str) -> Optional[Model]:
        """Load model from disk"""
        try:
            if os.path.exists(model_path):
                return load_model(model_path)
            return None
            
        except Exception as e:
            logger.error(f"Error loading model from {model_path}: {str(e)}")
            return None

    @staticmethod
    def save_model(model: Model, model_path: str) -> bool:
        """Save model to disk"""
        try:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            model.save(model_path)
            return True
            
        except Exception as e:
            logger.error(f"Error saving model to {model_path}: {str(e)}")
            return False

    @staticmethod
    def set_model_weights(model: Model, weights) -> Model:
        """Set model weights"""
        try:
            if weights is not None:
                model.set_weights(weights)
            return model
            
        except Exception as e:
            logger.error(f"Error setting model weights: {str(e)}")
            return model
