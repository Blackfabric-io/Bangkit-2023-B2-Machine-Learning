"""Base module containing the core model implementation."""

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

class CatDogModel:
    """Convolutional neural network model for cat/dog classification."""
    
    def __init__(self, input_shape: Tuple[int, int, int] = (150, 150, 3)) -> None:
        """Initialize the model architecture.
        
        Args:
            input_shape: Shape of input images (height, width, channels)
        """
        self.input_shape = input_shape
        self.model = tf.keras.models.Sequential([
            # First convolution
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            tf.keras.layers.MaxPooling2D(2, 2),
            
            # Second convolution
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            
            # Third convolution
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            
            # Fourth convolution
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            
            # Fifth convolution
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            
            # Flatten and dense layers
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        self.model.compile(
            optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
    def train(self, 
             train_generator: tf.keras.preprocessing.image.DirectoryIterator,
             validation_generator: Optional[tf.keras.preprocessing.image.DirectoryIterator] = None,
             epochs: int = 15,
             callback_threshold: float = 0.99,
             verbose: int = 1) -> tf.keras.callbacks.History:
        """Train the model with early stopping based on accuracy.
        
        Args:
            train_generator: Generator for training data
            validation_generator: Optional generator for validation data
            epochs: Number of training epochs
            callback_threshold: Accuracy threshold to stop training
            verbose: Verbosity mode
            
        Returns:
            Training history
        """
        if not isinstance(train_generator, tf.keras.preprocessing.image.DirectoryIterator):
            raise ValueError("train_generator must be a DirectoryIterator")
            
        callback = AccuracyCallback(threshold=callback_threshold)
        return self.model.fit(
            train_generator,
            validation_data=validation_generator,
            epochs=epochs,
            callbacks=[callback],
            verbose=verbose
        )
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict class for input images.
        
        Args:
            x: Input images (shape: [N, height, width, channels])
            
        Returns:
            Predicted probabilities (1 for dog, 0 for cat)
        """
        if not isinstance(x, np.ndarray):
            raise ValueError("Input must be a numpy array")
            
        if len(x.shape) != 4 or x.shape[1:] != self.input_shape:
            raise ValueError(f"Input images must have shape [N, {self.input_shape[0]}, {self.input_shape[1]}, {self.input_shape[2]}]")
            
        return self.model.predict(x)
    
    def evaluate(self, 
                test_generator: tf.keras.preprocessing.image.DirectoryIterator) -> Tuple[float, float]:
        """Evaluate model performance.
        
        Args:
            test_generator: Generator for test data
            
        Returns:
            Tuple of (loss, accuracy)
        """
        if not isinstance(test_generator, tf.keras.preprocessing.image.DirectoryIterator):
            raise ValueError("test_generator must be a DirectoryIterator")
            
        return self.model.evaluate(test_generator)
    
    def save(self, filepath: str) -> None:
        """Save model weights to file.
        
        Args:
            filepath: Path to save model weights
        """
        self.model.save_weights(filepath)
        
    def load(self, filepath: str) -> None:
        """Load model weights from file.
        
        Args:
            filepath: Path to load model weights from
        """
        self.model.load_weights(filepath) 