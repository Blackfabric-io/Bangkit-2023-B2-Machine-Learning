"""Data loading and preprocessing utilities for digit recognition.

This module handles:
- Loading MNIST dataset
- Data preprocessing and normalization
- Train/test splitting
- Batch generation
"""

import numpy as np
import tensorflow as tf
from typing import Tuple, Generator, Optional
import logging

logger = logging.getLogger(__name__)

class DataLoader:
    """Data loader for digit recognition dataset."""
    
    def __init__(self, flatten: bool = True, normalize: bool = True):
        """Initialize data loader.
        
        Args:
            flatten: Whether to flatten images to 1D arrays
            normalize: Whether to normalize pixel values to [0,1]
        """
        self.flatten = flatten
        self.normalize = normalize
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        
    def load_mnist(self) -> None:
        """Load MNIST dataset using TensorFlow."""
        logger.info("Loading MNIST dataset...")
        try:
            (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
            
            if self.flatten:
                X_train = X_train.reshape(X_train.shape[0], -1).T
                X_test = X_test.reshape(X_test.shape[0], -1).T
            
            if self.normalize:
                X_train = X_train / 255.0
                X_test = X_test / 255.0
            
            # Convert labels to one-hot encoding
            self.y_train = tf.keras.utils.to_categorical(y_train).T
            self.y_test = tf.keras.utils.to_categorical(y_test).T
            
            self.X_train = X_train
            self.X_test = X_test
            
            logger.info(f"Dataset loaded. Training samples: {X_train.shape[1]}, Test samples: {X_test.shape[1]}")
            
        except Exception as e:
            logger.error(f"Error loading MNIST dataset: {str(e)}")
            raise
    
    def get_batch(self, batch_size: int) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """Generate random batches of data.
        
        Args:
            batch_size: Size of each batch
            
        Yields:
            Tuple of (X_batch, y_batch)
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError("Dataset not loaded. Call load_mnist() first.")
            
        num_samples = self.X_train.shape[1]
        indices = np.arange(num_samples)
        
        while True:
            np.random.shuffle(indices)
            for i in range(0, num_samples, batch_size):
                batch_indices = indices[i:min(i + batch_size, num_samples)]
                yield self.X_train[:, batch_indices], self.y_train[:, batch_indices]
    
    def get_test_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get test dataset.
        
        Returns:
            Tuple of (X_test, y_test)
        """
        if self.X_test is None or self.y_test is None:
            raise ValueError("Dataset not loaded. Call load_mnist() first.")
        return self.X_test, self.y_test
    
    def get_train_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get training dataset.
        
        Returns:
            Tuple of (X_train, y_train)
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError("Dataset not loaded. Call load_mnist() first.")
        return self.X_train, self.y_train 