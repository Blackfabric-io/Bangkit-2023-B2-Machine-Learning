"""MNIST Digit Classification package.

A TensorFlow implementation of MNIST digit classification with accuracy-based early stopping.
"""

from .core import MNISTModel, AccuracyCallback
from .utils import load_mnist_data, normalize_images

__version__ = "0.1.0"
__all__ = ['MNISTModel', 'AccuracyCallback', 'load_mnist_data', 'normalize_images'] 