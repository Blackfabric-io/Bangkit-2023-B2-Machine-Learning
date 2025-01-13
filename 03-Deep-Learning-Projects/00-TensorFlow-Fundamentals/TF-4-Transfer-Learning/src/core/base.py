"""Core module containing model and callback implementations."""

import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.callbacks import Callback

class AccuracyThresholdCallback(Callback):
    """Custom callback that stops training when accuracy threshold is reached.
    
    Args:
        threshold (float): Accuracy threshold to stop training at (default: 0.999)
    """
    def __init__(self, threshold=0.999):
        super().__init__()
        self.threshold = threshold
        
    def on_epoch_end(self, epoch, logs=None):
        """Check accuracy at epoch end and stop if threshold reached."""
        if logs is None:
            logs = {}
        acc = logs.get('accuracy')
        if acc is not None and acc >= self.threshold:
            print(f'\nReached {self.threshold*100}% accuracy, stopping training.')
            self.model.stop_training = True

class EmotionModel:
    """CNN model for binary classification of happy/sad images.
    
    The model architecture consists of:
    - 3 Conv2D layers with ReLU activation
    - 3 MaxPooling2D layers
    - Dense layer with ReLU activation
    - Output layer with sigmoid activation
    """
    def __init__(self):
        self.model = Sequential([
            Conv2D(16, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            MaxPooling2D(2, 2),
            Conv2D(16, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Conv2D(16, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Flatten(),
            Dense(512, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        self.model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
    def train(self, train_generator, epochs=15, callbacks=None):
        """Train the model on image data.
        
        Args:
            train_generator: Keras ImageDataGenerator for training data
            epochs (int): Maximum number of training epochs
            callbacks (list): List of Keras callbacks (optional)
            
        Returns:
            History object containing training metrics
        """
        if callbacks is None:
            callbacks = [AccuracyThresholdCallback()]
            
        return self.model.fit(
            train_generator,
            epochs=epochs,
            callbacks=callbacks
        )
    
    def predict(self, image):
        """Make prediction on a single preprocessed image.
        
        Args:
            image (numpy.ndarray): Preprocessed image array of shape (28, 28, 1)
            
        Returns:
            float: Probability of "happy" class (0-1)
            
        Raises:
            ValueError: If image shape is invalid
        """
        if not isinstance(image, np.ndarray):
            raise ValueError("Input must be a numpy array")
            
        if image.shape != (28, 28, 1):
            raise ValueError(f"Expected shape (28, 28, 1), got {image.shape}")
            
        # Add batch dimension
        image = np.expand_dims(image, axis=0)
        return float(self.model.predict(image)[0, 0])
    
    def save(self, filepath):
        """Save model weights to file.
        
        Args:
            filepath (str): Path to save model weights
        """
        self.model.save_weights(filepath)
        
    def load(self, filepath):
        """Load model weights from file.
        
        Args:
            filepath (str): Path to load model weights from
        """
        self.model.load_weights(filepath) 