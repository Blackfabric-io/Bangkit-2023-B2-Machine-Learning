"""Core model implementations for stock price prediction."""

from typing import Tuple, Optional, Dict, Any
import numpy as np
import numpy.typing as npt
import tensorflow as tf
from tensorflow import keras

class BaseModel:
    """Base class for stock price prediction models."""
    
    def __init__(self,
                 sequence_length: int = 60,
                 n_features: int = 1,
                 batch_size: int = 32) -> None:
        """Initialize base model.
        
        Args:
            sequence_length: Number of time steps in each sequence
            n_features: Number of input features
            batch_size: Training batch size
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.batch_size = batch_size
        self.model = None
        
    def build_model(self) -> None:
        """Build model architecture."""
        raise NotImplementedError("Subclasses must implement build_model")
        
    def compile_model(self,
                     learning_rate: float = 0.001) -> None:
        """Compile model with optimizer and loss function.
        
        Args:
            learning_rate: Learning rate for optimizer
        """
        if self.model is None:
            self.build_model()
            
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='mean_squared_error'
        )
        
    def train(self,
             X_train: npt.NDArray[np.float32],
             y_train: npt.NDArray[np.float32],
             X_val: npt.NDArray[np.float32],
             y_val: npt.NDArray[np.float32],
             epochs: int = 50,
             patience: int = 10) -> Dict[str, Any]:
        """Train the model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            epochs: Number of training epochs
            patience: Early stopping patience
            
        Returns:
            Training history
        """
        if self.model is None:
            self.compile_model()
            
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            mode='min',
            restore_best_weights=True
        )
        
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=self.batch_size,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping],
            verbose=1
        )
        
        return history.history
    
    def predict(self, X: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Make predictions.
        
        Args:
            X: Input features
            
        Returns:
            Model predictions
            
        Raises:
            ValueError: If model hasn't been trained
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
            
        return self.model.predict(X)

class RNNModel(BaseModel):
    """Simple RNN model for stock price prediction."""
    
    def build_model(self) -> None:
        """Build RNN model architecture."""
        self.model = tf.keras.Sequential([
            tf.keras.layers.SimpleRNN(50, input_shape=(self.sequence_length, self.n_features)),
            tf.keras.layers.Dense(1)
        ])

class LSTMModel(BaseModel):
    """LSTM model for stock price prediction."""
    
    def build_model(self) -> None:
        """Build LSTM model architecture."""
        self.model = tf.keras.Sequential([
            tf.keras.layers.LSTM(50, input_shape=(self.sequence_length, self.n_features)),
            tf.keras.layers.Dense(1)
        ])

class BiLSTMModel(BaseModel):
    """Bidirectional LSTM model for stock price prediction."""
    
    def build_model(self) -> None:
        """Build Bidirectional LSTM model architecture."""
        self.model = tf.keras.Sequential([
            tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(50),
                input_shape=(self.sequence_length, self.n_features)
            ),
            tf.keras.layers.Dense(1)
        ]) 