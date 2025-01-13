"""Helper functions for data loading and preprocessing."""

import os
from typing import Tuple
import tensorflow as tf
import numpy as np

def load_mnist_data(data_path: str = None) -> Tuple[np.ndarray, np.ndarray]:
    """Load MNIST training data.
    
    Args:
        data_path: Optional path to mnist.npz file
        
    Returns:
        Tuple of (images, labels)
    """
    try:
        if data_path is None:
            # Use default path
            (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
        else:
            # Use provided path
            (x_train, y_train), _ = tf.keras.datasets.mnist.load_data(path=data_path)
            
        return x_train, y_train
    except Exception as e:
        raise RuntimeError(f"Failed to load MNIST data: {str(e)}")

def preprocess_images(images: np.ndarray) -> np.ndarray:
    """Preprocess images by adding channel dimension and normalizing.
    
    Args:
        images: Input images with shape [N, 28, 28]
        
    Returns:
        Preprocessed images with shape [N, 28, 28, 1] and values in [0, 1]
    """
    if not isinstance(images, np.ndarray):
        raise ValueError("Input must be a numpy array")
        
    if len(images.shape) != 3 or images.shape[1:] != (28, 28):
        raise ValueError("Input images must have shape [N, 28, 28]")
        
    # Add channel dimension and normalize
    images = np.expand_dims(images, axis=-1)
    images = images.astype('float32') / 255.0
    
    return images 