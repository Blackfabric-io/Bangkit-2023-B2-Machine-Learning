"""Happy/Sad Image Classification package.

A TensorFlow implementation of binary image classification using CNN with accuracy-based early stopping.
"""

from .core import EmotionModel, AccuracyThresholdCallback
from .utils import create_data_generator, load_and_preprocess_image

__version__ = "0.1.0"
__all__ = ['EmotionModel', 'AccuracyThresholdCallback', 'create_data_generator', 'load_and_preprocess_image'] 