"""
Core FashionMNIST model implementation with TensorBoard visualization support.
"""

from typing import List, Tuple, Optional
import os
import logging
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FashionMNISTModel:
    """
    FashionMNIST CNN model with TensorBoard support.
    """
    
    CLASS_NAMES = [
        'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
    ]
    
    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (28, 28, 1),
        num_classes: int = 10
    ) -> None:
        """
        Initialize the FashionMNIST model.
        
        Args:
            input_shape: Shape of input images (height, width, channels)
            num_classes: Number of output classes
        """
        self._input_shape = input_shape
        self._num_classes = num_classes
        self._model = self._build_model()
        
    def _build_model(self) -> tf.keras.Model:
        """Build the CNN model architecture."""
        try:
            model = tf.keras.models.Sequential([
                tf.keras.layers.Conv2D(64, (3,3), activation='relu', 
                                     input_shape=self._input_shape),
                tf.keras.layers.MaxPooling2D(2, 2),
                tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
                tf.keras.layers.MaxPooling2D(2,2),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dense(self._num_classes, activation='softmax')
            ])
            
            model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            logger.info("Model built successfully")
            return model
            
        except Exception as e:
            logger.error(f"Error building model: {str(e)}")
            raise
            
    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load and preprocess the Fashion MNIST dataset.
        
        Returns:
            Tuple of (train_images, train_labels, test_images, test_labels)
        """
        try:
            # Get current working directory for data storage
            data_dir = os.path.join(os.getcwd(), 'tensorflow_datasets')
            
            # Load data from numpy files
            train_images = np.load(f"{data_dir}/train_images.npy")
            train_labels = np.load(f"{data_dir}/train_labels.npy")
            test_images = np.load(f"{data_dir}/test_images.npy")
            test_labels = np.load(f"{data_dir}/test_labels.npy")
            
            # Reshape and normalize images
            train_images = train_images.reshape(-1, *self._input_shape) / 255.0
            test_images = test_images.reshape(-1, *self._input_shape) / 255.0
            
            logger.info("Data loaded and preprocessed successfully")
            return train_images, train_labels, test_images, test_labels
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
            
    def train(
        self,
        epochs: int = 5,
        batch_size: int = 32,
        validation_split: float = 0.2,
        callbacks: Optional[List[tf.keras.callbacks.Callback]] = None
    ) -> tf.keras.callbacks.History:
        """
        Train the model on the Fashion MNIST dataset.
        
        Args:
            epochs: Number of training epochs
            batch_size: Training batch size
            validation_split: Fraction of data to use for validation
            callbacks: List of Keras callbacks for training
            
        Returns:
            Training history
        """
        try:
            # Load and preprocess data
            train_images, train_labels, test_images, test_labels = self.load_data()
            
            # Train the model
            history = self._model.fit(
                train_images,
                train_labels,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                callbacks=callbacks
            )
            
            logger.info("Model training completed")
            return history
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise
            
    def evaluate(
        self,
        batch_size: int = 32
    ) -> Tuple[float, float]:
        """
        Evaluate the model on the test dataset.
        
        Args:
            batch_size: Evaluation batch size
            
        Returns:
            Tuple of (loss, accuracy)
        """
        try:
            # Load test data
            _, _, test_images, test_labels = self.load_data()
            
            # Evaluate model
            results = self._model.evaluate(
                test_images,
                test_labels,
                batch_size=batch_size
            )
            
            # Log results
            for name, value in zip(self._model.metrics_names, results):
                logger.info(f"{name}: {value:.3f}")
                
            return results
            
        except Exception as e:
            logger.error(f"Error during evaluation: {str(e)}")
            raise
            
    def predict(
        self,
        images: np.ndarray,
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Make predictions on input images.
        
        Args:
            images: Input images of shape [batch_size, height, width, channels]
            batch_size: Prediction batch size
            
        Returns:
            Predicted class probabilities
        """
        try:
            predictions = self._model.predict(
                images,
                batch_size=batch_size
            )
            return predictions
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise 