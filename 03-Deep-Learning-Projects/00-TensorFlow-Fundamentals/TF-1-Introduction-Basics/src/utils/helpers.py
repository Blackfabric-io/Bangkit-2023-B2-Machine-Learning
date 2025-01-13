"""Helper functions for data preprocessing and postprocessing."""

import numpy as np
from typing import Union, List

def scale_price(price: Union[float, List[float], np.ndarray]) -> np.ndarray:
    """Scale house prices from thousands to normalized values (1.0 = 100k).
    
    Args:
        price: Raw price in thousands (e.g., 150 for 150k)
        
    Returns:
        Scaled price (e.g., 1.5 for 150k)
    """
    if isinstance(price, (list, np.ndarray)):
        return np.array(price) / 100.0
    return np.array([price / 100.0])

def unscale_price(scaled_price: Union[float, List[float], np.ndarray]) -> np.ndarray:
    """Convert scaled prices back to thousands.
    
    Args:
        scaled_price: Normalized price (e.g., 1.5 for 150k)
        
    Returns:
        Price in thousands (e.g., 150 for 150k)
    """
    if isinstance(scaled_price, (list, np.ndarray)):
        return np.array(scaled_price) * 100.0
    return np.array([scaled_price * 100.0]) 