"""Base module containing the core model and callback implementations."""

from typing import Tuple, Optional, Dict, Any
import tensorflow as tf
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AccuracyThresholdCallback(tf.keras.callbacks.Callback):
    """Callback to stop training when accuracy threshold is reached."""
    
    def __init__(self, threshold: float = 0.995) -> None:
        """Initialize the callback.
        
        Args:
            threshold: Accuracy threshold to stop training at (default: 0.995 for 99.5%)
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

class CNNModel:
    """Convolutional neural network model for MNIST digit classification."""
    
    def __init__(self) -> None:
        """Initialize the model architecture."""
        self.model = tf.keras.models.Sequential([
            # Convolutional layer
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            tf.keras.layers.MaxPooling2D(2, 2),
            
            # Flatten and dense layers
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation=tf.nn.relu),
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
             callback_threshold: float = 0.995,
             verbose: int = 1) -> tf.keras.callbacks.History:
        """Train the model with early stopping based on accuracy.
        
        Args:
            x_train: Training images (shape: [N, 28, 28, 1])
            y_train: Training labels
            epochs: Number of training epochs
            callback_threshold: Accuracy threshold to stop training
            verbose: Verbosity mode
            
        Returns:
            Training history
        """
        # Validate input shapes
        if not isinstance(x_train, np.ndarray) or not isinstance(y_train, np.ndarray):
            raise ValueError("Training data must be numpy arrays")
            
        if len(x_train.shape) != 4 or x_train.shape[1:] != (28, 28, 1):
            raise ValueError("Input images must have shape [N, 28, 28, 1]")
            
        if x_train.shape[0] != y_train.shape[0]:
            raise ValueError("Number of samples and labels must match")
            
        callback = AccuracyThresholdCallback(threshold=callback_threshold)
        return self.model.fit(x_train, y_train, epochs=epochs, callbacks=[callback], verbose=verbose)
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict digit classes for input images.
        
        Args:
            x: Input images (shape: [N, 28, 28, 1])
            
        Returns:
            Predicted probabilities for each digit class
        """
        if not isinstance(x, np.ndarray):
            raise ValueError("Input must be a numpy array")
            
        if len(x.shape) != 4 or x.shape[1:] != (28, 28, 1):
            raise ValueError("Input images must have shape [N, 28, 28, 1]")
            
        return self.model.predict(x)
    
    def evaluate(self, x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """Evaluate model performance.
        
        Args:
            x: Test images (shape: [N, 28, 28, 1])
            y: Test labels
            
        Returns:
            Tuple of (loss, accuracy)
        """
        if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
            raise ValueError("Test data must be numpy arrays")
            
        if len(x.shape) != 4 or x.shape[1:] != (28, 28, 1):
            raise ValueError("Input images must have shape [N, 28, 28, 1]")
            
        return self.model.evaluate(x, y) 