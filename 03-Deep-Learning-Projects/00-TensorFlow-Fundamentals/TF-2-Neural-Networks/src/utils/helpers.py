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

def normalize_images(images: np.ndarray) -> np.ndarray:
    """Normalize image pixel values to [0, 1] range.
    
    Args:
        images: Input images with pixel values in [0, 255]
        
    Returns:
        Normalized images with pixel values in [0, 1]
    """
    if not isinstance(images, np.ndarray):
        raise ValueError("Input must be a numpy array")
        
    if images.dtype != np.uint8 and not np.issubdtype(images.dtype, np.floating):
        raise ValueError("Input array must be uint8 or float")
        
    # Convert to float and normalize
    return images.astype('float32') / 255.0 