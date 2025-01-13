"""TensorFlow House Price Prediction package.

A simple package demonstrating TensorFlow basics through house price prediction.
"""

from .core import HousePriceModel
from .utils import scale_price, unscale_price

__version__ = "0.1.0"
__all__ = ['HousePriceModel', 'scale_price', 'unscale_price'] 