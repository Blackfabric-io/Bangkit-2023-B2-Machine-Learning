"""Base module containing the core model and callback implementations."""

from typing import Tuple, Optional, Dict, Any
import tensorflow as tf
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AccuracyCallback(tf.keras.callbacks.Callback):
    """Callback to stop training when accuracy threshold is reached."""
    
    def __init__(self, threshold: float = 0.99) -> None:
        """Initialize the callback.
        
        Args:
            threshold: Accuracy threshold to stop training at
        """
        super().__init__()
        self.threshold = threshold
        
    def on_epoch_end(self, epoch: int, logs: Dict[str, Any] = None) -> None:
        """Check accuracy at epoch end and stop if threshold is reached.
        
        Args:
            epoch: Current epoch number
            logs: Training metrics
        """
        logs = logs or {}
        accuracy = logs.get('accuracy')
        
        if accuracy is None:
            logger.warning("Accuracy metric not found in training logs")
            return
            
        if accuracy > self.threshold:
            logger.info(f"\nReached {self.threshold*100}% accuracy so cancelling training!")
            self.model.stop_training = True

class MNISTModel:
    """Neural network model for MNIST digit classification."""
    
    def __init__(self) -> None:
        """Initialize the model architecture."""
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(512, activation=tf.nn.relu),
            tf.keras.layers.Dense(10, activation=tf.nn.softmax)
        ])
        
        self.model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
    def train(self, 
             x_train: np.ndarray, 
             y_train: np.ndarray,
             epochs: int = 10,
             callback_threshold: float = 0.99,
             verbose: int = 1) -> tf.keras.callbacks.History:
        """Train the model with early stopping based on accuracy.
        
        Args:
            x_train: Training images
            y_train: Training labels
            epochs: Number of training epochs
            callback_threshold: Accuracy threshold to stop training
            verbose: Verbosity mode
            
        Returns:
            Training history
        """
        if not isinstance(x_train, np.ndarray) or not isinstance(y_train, np.ndarray):
            raise ValueError("Training data must be numpy arrays")
            
        if x_train.shape[0] != y_train.shape[0]:
            raise ValueError("Number of samples and labels must match")
            
        callback = AccuracyCallback(threshold=callback_threshold)
        return self.model.fit(x_train, y_train, epochs=epochs, callbacks=[callback], verbose=verbose)
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict digit classes for input images.
        
        Args:
            x: Input images
            
        Returns:
            Predicted probabilities for each digit class
        """
        if not isinstance(x, np.ndarray):
            raise ValueError("Input must be a numpy array")
            
        return self.model.predict(x)
    
    def evaluate(self, x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """Evaluate model performance.
        
        Args:
            x: Test images
            y: Test labels
            
        Returns:
            Tuple of (loss, accuracy)
        """
        if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
            raise ValueError("Test data must be numpy arrays")
            
        return self.model.evaluate(x, y) 