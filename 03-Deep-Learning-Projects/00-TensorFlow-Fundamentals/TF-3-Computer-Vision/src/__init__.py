"""MNIST CNN Classification package.

A TensorFlow implementation of MNIST digit classification using CNN with accuracy-based early stopping.
"""

from .core import CNNModel, AccuracyThresholdCallback
from .utils import load_mnist_data, preprocess_images

__version__ = "0.1.0"
__all__ = ['CNNModel', 'AccuracyThresholdCallback', 'load_mnist_data', 'preprocess_images'] 