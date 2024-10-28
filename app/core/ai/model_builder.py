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
        self.epsilon = 0.1

    def build_lstm_model(self, input_shape: Tuple[int, int]) -> Model:
        """Build LSTM model with specified input shape"""
        try:
            model = Sequential([
                LSTM(128, return_sequences=True, input_shape=input_shape),
                Dropout(0.3),
                LSTM(64, return_sequences=True),
                Dropout(0.2),
                LSTM(32),
                Dropout(0.2),
                Dense(16, activation='relu'),
                Dense(self.prediction_horizon)  # Output layer matches prediction horizon
            ])
            
            model.compile(
                optimizer=Adam(learning_rate=self.learning_rate),
                loss='huber',
                metrics=['mae', 'mse']
            )
            
            return model
            
        except Exception as e:
            logger.error(f"Error building LSTM model: {str(e)}")
            raise

    def prepare_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for training"""
        try:
            # Verify input is DataFrame
            if not isinstance(data, pd.DataFrame):
                raise ValueError(f"Expected DataFrame, got {type(data)}")
                
            if data.empty:
                raise ValueError("Empty DataFrame provided")
                
            # Ensure required columns exist
            required_columns = ['close', 'volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Scale the data
            scaled_data = self.scaler.fit_transform(data[['close', 'volume']].values)
            
            # Create sequences
            X, y = [], []
            for i in range(len(scaled_data) - self.sequence_length - self.prediction_horizon + 1):
                # Input sequence
                X.append(scaled_data[i:(i + self.sequence_length)])
                # Target sequence (next prediction_horizon close prices)
                y.append(scaled_data[i + self.sequence_length:i + self.sequence_length + self.prediction_horizon, 0])
            
            return np.array(X), np.array(y)
            
        except Exception as e:
            logger.error(f"Error preparing data: {str(e)}")
            raise

    def train_model(self, data_feed: pd.DataFrame) -> TrainingResult:
        """Train the model with reinforcement learning"""
        try:
            # Ensure minimum data length
            min_length = self.sequence_length + self.prediction_horizon + 1
            if len(data_feed) < min_length:
                raise ValueError(f"Insufficient data length. Need at least {min_length} points.")

            # Split data into train and validation
            train_size = int(len(data_feed) * 0.8)
            train_data = data_feed.iloc[:train_size]
            eval_data = data_feed.iloc[train_size:]
            
            # Prepare data
            X_train, y_train = self.prepare_data(train_data)
            X_val, y_val = self.prepare_data(eval_data)
            
            # Build model
            input_shape = (self.sequence_length, X_train.shape[2])
            model = self.build_lstm_model(input_shape)
            
            # Train model
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=self.epochs,
                batch_size=self.batch_size,
                verbose=1
            )
            
            # Calculate metrics
            performance_metrics = {
                'mse': float(history.history['loss'][-1]),
                'mae': float(history.history['mae'][-1]),
                'val_mse': float(history.history['val_loss'][-1]),
                'val_mae': float(history.history['val_mae'][-1])
            }
            
            validation_metrics = self._calculate_validation_metrics(
                model, X_val, y_val, eval_data
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
            
            # Reshape predictions and actual values for inverse transform
            pred_shape = predictions.shape
            y_shape = y_val.shape
            
            predictions_reshaped = predictions.reshape(-1, 1)
            y_val_reshaped = y_val.reshape(-1, 1)
            
            # Add dummy column for volume (since we scaled with both close and volume)
            pred_with_volume = np.column_stack((predictions_reshaped, np.zeros_like(predictions_reshaped)))
            y_with_volume = np.column_stack((y_val_reshaped, np.zeros_like(y_val_reshaped)))
            
            # Inverse transform
            unscaled_predictions = self.scaler.inverse_transform(pred_with_volume)[:, 0].reshape(pred_shape)
            unscaled_actual = self.scaler.inverse_transform(y_with_volume)[:, 0].reshape(y_shape)
            
            # Calculate metrics
            mse = np.mean((unscaled_predictions - unscaled_actual) ** 2)
            mae = np.mean(np.abs(unscaled_predictions - unscaled_actual))
            
            # Calculate directional accuracy
            pred_direction = np.diff(unscaled_predictions.mean(axis=1)) > 0
            actual_direction = np.diff(unscaled_actual.mean(axis=1)) > 0
            directional_accuracy = np.mean(pred_direction == actual_direction)
            
            return {
                'mse': float(mse),
                'mae': float(mae),
                'directional_accuracy': float(directional_accuracy),
                'average_prediction_error': float(mae / np.mean(np.abs(unscaled_actual)))
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

    def build_rl_model(self, input_shape: Tuple[int], action_size: int) -> Model:
        """Build model for reinforcement learning"""
        try:
            model = Sequential([
                Dense(64, activation='relu', input_shape=input_shape),
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

    def train_on_batch(self, model: Model, states: np.ndarray, targets: np.ndarray) -> float:
        """Train model on a single batch of data"""
        try:
            history = model.train_on_batch(states, targets)
            return float(history[0])  # Return loss value
        except Exception as e:
            logger.error(f"Error training on batch: {str(e)}")
            raise

