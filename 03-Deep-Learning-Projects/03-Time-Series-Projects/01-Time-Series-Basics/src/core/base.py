"""Core time series forecasting functionality."""

from typing import Tuple, Union, Optional
import numpy as np
import numpy.typing as npt
from matplotlib import pyplot as plt

def trend(time: npt.NDArray[np.float32], slope: float = 0) -> npt.NDArray[np.float32]:
    """Generate a linear trend over time.
    
    Args:
        time: Array of time points
        slope: Slope of the trend line
        
    Returns:
        Array containing trend values
    """
    return slope * time

def seasonal_pattern(season_time: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Generate an arbitrary seasonal pattern.
    
    Args:
        season_time: Normalized time within season (0 to 1)
        
    Returns:
        Array containing pattern values
    """
    return np.where(season_time < 0.1,
                   np.cos(season_time * 7 * np.pi),
                   1 / np.exp(5 * season_time))

def seasonality(time: npt.NDArray[np.float32], 
               period: float,
               amplitude: float = 1,
               phase: float = 0) -> npt.NDArray[np.float32]:
    """Generate repeating seasonality pattern.
    
    Args:
        time: Array of time points
        period: Length of seasonal period
        amplitude: Magnitude of seasonality
        phase: Phase shift of pattern
        
    Returns:
        Array containing seasonal values
    """
    season_time = ((time + phase) % period) / period
    return amplitude * seasonal_pattern(season_time)

def noise(time: npt.NDArray[np.float32],
         noise_level: float = 1,
         seed: Optional[int] = None) -> npt.NDArray[np.float32]:
    """Generate random noise.
    
    Args:
        time: Array of time points (used for length)
        noise_level: Standard deviation of noise
        seed: Random seed for reproducibility
        
    Returns:
        Array containing noise values
    """
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time)) * noise_level

def plot_series(time: npt.NDArray[np.float32],
               series: npt.NDArray[np.float32],
               format: str = "-",
               title: str = "",
               label: Optional[str] = None,
               start: int = 0,
               end: Optional[int] = None) -> None:
    """Plot a time series.
    
    Args:
        time: Array of time points
        series: Array of values
        format: Plot format string
        title: Plot title
        label: Legend label
        start: Start index for plotting
        end: End index for plotting
    """
    plt.plot(time[start:end], series[start:end], format, label=label)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.title(title)
    if label:
        plt.legend()
    plt.grid(True)

def train_val_split(time: npt.NDArray[np.float32],
                   series: npt.NDArray[np.float32],
                   time_step: int = 1100) -> Tuple[npt.NDArray[np.float32], 
                                                 npt.NDArray[np.float32],
                                                 npt.NDArray[np.float32],
                                                 npt.NDArray[np.float32]]:
    """Split time series into training and validation sets.
    
    Args:
        time: Array of time points
        series: Array of values
        time_step: Index at which to split
        
    Returns:
        Tuple containing:
            - Training time points
            - Training values
            - Validation time points
            - Validation values
    """
    time_train = time[:time_step]
    series_train = series[:time_step]
    time_valid = time[time_step:]
    series_valid = series[time_step:]
    
    return time_train, series_train, time_valid, series_valid

def compute_metrics(true_series: npt.NDArray[np.float32],
                   forecast: npt.NDArray[np.float32]) -> Tuple[float, float]:
    """Compute evaluation metrics for forecasts.
    
    Args:
        true_series: Ground truth values
        forecast: Predicted values
        
    Returns:
        Tuple containing:
            - Mean squared error
            - Mean absolute error
    """
    mse = np.square(forecast - true_series).mean()
    mae = np.abs(forecast - true_series).mean()
    return mse, mae 