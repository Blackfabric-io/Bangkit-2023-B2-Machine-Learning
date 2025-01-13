"""Cats vs Dogs Classification package.

A TensorFlow implementation of binary image classification using CNN with data augmentation.
"""

from .core import CatDogModel, AccuracyCallback
from .utils import create_data_generators, load_and_preprocess_image

__version__ = "0.1.0"
__all__ = ['CatDogModel', 'AccuracyCallback', 'create_data_generators', 'load_and_preprocess_image'] 