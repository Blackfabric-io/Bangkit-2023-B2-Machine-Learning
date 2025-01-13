"""Base module containing the core HousePriceModel implementation."""

from typing import Tuple, Optional
import tensorflow as tf
import numpy as np

class HousePriceModel:
    """A simple neural network model for predicting house prices based on number of bedrooms."""
    
    def __init__(self) -> None:
        """Initialize the model with a single dense layer."""
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(units=1, input_shape=[1])
        ])
        self.model.compile(optimizer='sgd', loss='mean_squared_error')
        
    def train(self, 
              bedrooms: np.ndarray, 
              prices: np.ndarray, 
              epochs: int = 1000,
              verbose: int = 1) -> tf.keras.callbacks.History:
        """Train the model on bedroom counts and corresponding prices.
        
        Args:
            bedrooms: Array of bedroom counts
            prices: Array of corresponding house prices (scaled)
            epochs: Number of training epochs
            verbose: Verbosity mode
            
        Returns:
            Training history
        """
        return self.model.fit(bedrooms, prices, epochs=epochs, verbose=verbose)
    
    def predict(self, bedrooms: np.ndarray) -> np.ndarray:
        """Predict house prices for given bedroom counts.
        
        Args:
            bedrooms: Array of bedroom counts
            
        Returns:
            Predicted house prices (scaled)
        """
        return self.model.predict(bedrooms)
    
    @staticmethod
    def prepare_data() -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data with scaled prices.
        
        Returns:
            Tuple of (bedroom counts, scaled prices)
        """
        bedrooms = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=float)
        # Scale prices down (1.0 = 100k)
        prices = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5], dtype=float)
        return bedrooms, prices 