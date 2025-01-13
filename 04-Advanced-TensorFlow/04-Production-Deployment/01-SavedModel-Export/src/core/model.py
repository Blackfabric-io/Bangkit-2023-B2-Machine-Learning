"""
Core MNIST model implementation with SavedModel export functionality.
"""

from typing import Tuple, Optional
import os
import logging
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MNIST:
    """
    MNIST model class that handles training, evaluation and SavedModel export.
    """
    
    def __init__(
        self,
        export_path: str,
        buffer_size: int = 1000,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        epochs: int = 10
    ) -> None:
        """
        Initialize the MNIST model.
        
        Args:
            export_path: Path where the SavedModel will be exported
            buffer_size: Size of the shuffle buffer for training
            batch_size: Batch size for training and evaluation
            learning_rate: Learning rate for the Adam optimizer
            epochs: Number of training epochs
        """
        self._export_path = export_path
        self._buffer_size = buffer_size
        self._batch_size = batch_size
        self._learning_rate = learning_rate
        self._epochs = epochs
        
        self._build_model()
        self.train_dataset, self.test_dataset = self._prepare_dataset()
    
    def preprocess_fn(self, x: tf.Tensor) -> tf.Tensor:
        """
        Preprocess input images.
        
        Args:
            x: Input tensor of shape [height, width, channels]
            
        Returns:
            Preprocessed tensor normalized to [0, 1]
        """
        try:
            x = tf.cast(x, tf.float32) / 255.0
            return x
        except Exception as e:
            logger.error(f"Error in preprocessing: {str(e)}")
            raise
            
    def _build_model(self) -> None:
        """Build the CNN model architecture."""
        try:
            self._model = tf.keras.models.Sequential([
                tf.keras.layers.Input(shape=(28, 28, 1), dtype=tf.uint8),
                tf.keras.layers.Lambda(self.preprocess_fn),
                tf.keras.layers.Conv2D(8, (3, 3), padding='same'),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Conv2D(16, (3, 3), padding='same'),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Conv2D(32, (3, 3), padding='same'),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128),
                tf.keras.layers.Dense(10, activation='softmax')
            ])
            
            optimizer = tf.keras.optimizers.Adam(learning_rate=self._learning_rate)
            self._model.compile(
                optimizer=optimizer,
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            logger.info("Model built successfully")
            
        except Exception as e:
            logger.error(f"Error building model: {str(e)}")
            raise
            
    def _prepare_dataset(self) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        """
        Prepare the MNIST training and test datasets.
        
        Returns:
            Tuple of (train_dataset, test_dataset)
        """
        try:
            # Get current working directory for data storage
            data_dir = os.path.join(os.getcwd(), 'tensorflow_datasets')
            
            # Load training data
            train_dataset = tfds.load(
                'mnist',
                split='train',
                data_dir=data_dir,
                as_supervised=True
            )
            
            # Load test data
            test_dataset = tfds.load(
                'mnist',
                split='test',
                data_dir=data_dir,
                as_supervised=True
            )
            
            logger.info("Datasets prepared successfully")
            return train_dataset, test_dataset
            
        except Exception as e:
            logger.error(f"Error preparing dataset: {str(e)}")
            raise
            
    def train(self) -> None:
        """Train the model on the MNIST dataset."""
        try:
            # Prepare training dataset
            dataset_tr = self.train_dataset.shuffle(
                self._buffer_size
            ).batch(self._batch_size)
            
            # Train the model
            self._model.fit(dataset_tr, epochs=self._epochs)
            logger.info("Model training completed")
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise
            
    def test(self) -> None:
        """Evaluate the model on the test dataset."""
        try:
            # Prepare test dataset
            dataset_te = self.test_dataset.batch(self._batch_size)
            
            # Evaluate model
            results = self._model.evaluate(dataset_te)
            
            # Log results
            for name, value in zip(self._model.metrics_names, results):
                logger.info(f"{name}: {value:.3f}")
                
        except Exception as e:
            logger.error(f"Error during testing: {str(e)}")
            raise
            
    def export_model(self) -> None:
        """Export the model in SavedModel format."""
        try:
            tf.saved_model.save(self._model, self._export_path)
            logger.info(f"Model exported successfully to {self._export_path}")
            
        except Exception as e:
            logger.error(f"Error exporting model: {str(e)}")
            raise
            
    def predict(self, images: np.ndarray) -> np.ndarray:
        """
        Make predictions on input images.
        
        Args:
            images: Input images of shape [batch_size, height, width, channels]
            
        Returns:
            Predicted class probabilities
        """
        try:
            predictions = self._model.predict(images)
            return predictions
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise 