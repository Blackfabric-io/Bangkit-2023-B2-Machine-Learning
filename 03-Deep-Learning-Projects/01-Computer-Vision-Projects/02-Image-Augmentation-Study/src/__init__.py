"""Image Augmentation Study package.

A comprehensive implementation of image augmentation techniques for deep learning.
"""

from .core import ImageAugmenter, AugmentationConfig
from .utils import load_image, save_image, visualize_augmentations

__version__ = "0.1.0"
__all__ = ['ImageAugmenter', 'AugmentationConfig', 'load_image', 'save_image', 'visualize_augmentations'] 