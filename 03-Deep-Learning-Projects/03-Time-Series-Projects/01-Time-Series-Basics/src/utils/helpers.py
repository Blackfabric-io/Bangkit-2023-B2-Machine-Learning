"""Helper functions for time series analysis."""

from typing import Tuple
import numpy as np
import numpy.typing as npt

def generate_time_series(n_points: int = 1461,
                        slope: float = 0.01,
                        y_intercept: float = 10,
                        period: int = 365,
                        amplitude: float = 40,
                        noise_level: float = 2,
                        seed: int = 42) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Generate synthetic time series data.
    
    Args:
        n_points: Number of time points
        slope: Trend slope
        y_intercept: Y-intercept
        period: Seasonality period
        amplitude: Seasonality amplitude
        noise_level: Noise standard deviation
        seed: Random seed
        
    Returns:
        Tuple containing:
            - Array of time points
            - Array of time series values
    """
    from ..core.base import trend, seasonality, noise
    
    # Generate time points
    time = np.arange(n_points, dtype=np.float32)
    
    # Generate series components
    series = trend(time, slope) + y_intercept
    series += seasonality(time, period=period, amplitude=amplitude)
    series += noise(time, noise_level=noise_level, seed=seed)
    
    return time, series

def smooth_series(series: npt.NDArray[np.float32],
                 window_size: int = 10) -> npt.NDArray[np.float32]:
    """Apply centered moving average smoothing.
    
    Args:
        series: Input time series
        window_size: Size of smoothing window
        
    Returns:
        Smoothed time series
    """
    from ..processors.main import moving_average_forecast
    
    # Pad series to handle edges
    pad_width = window_size // 2
    padded_series = np.pad(series, (pad_width, pad_width), mode='edge')
    
    # Apply moving average
    smoothed = moving_average_forecast(padded_series, window_size)
    
    # Remove padding effects
    return smoothed[pad_width:-pad_width] 