"""Main forecasting methods implementation."""

from typing import List, Optional
import numpy as np
import numpy.typing as npt

def moving_average_forecast(series: npt.NDArray[np.float32],
                          window_size: int) -> npt.NDArray[np.float32]:
    """Compute moving average forecast.
    
    Args:
        series: Input time series
        window_size: Size of moving average window
        
    Returns:
        Array containing moving average forecasts
    """
    forecast: List[float] = []
    
    for time in range(len(series) - window_size):
        forecast.append(series[time:time + window_size].mean())
    
    return np.array(forecast)

def naive_forecast(series: npt.NDArray[np.float32],
                  split_time: int) -> npt.NDArray[np.float32]:
    """Compute naive forecast (using previous value).
    
    Args:
        series: Input time series
        split_time: Time step to split at
        
    Returns:
        Array containing naive forecasts
    """
    return series[split_time-1:len(series)-1]

def diff_forecast(series: npt.NDArray[np.float32],
                 period: int = 365,
                 window_size: int = 50,
                 split_time: Optional[int] = None) -> npt.NDArray[np.float32]:
    """Compute differenced moving average forecast.
    
    Args:
        series: Input time series
        period: Differencing period
        window_size: Size of moving average window
        split_time: Time step to split at
        
    Returns:
        Array containing differenced moving average forecasts
    """
    # Compute differences
    diff_series = np.array([series[i] - series[i-period] 
                           for i in range(period, len(series))])
    
    # Apply moving average to differences
    diff_moving_avg = moving_average_forecast(diff_series, window_size)
    
    if split_time is not None:
        # Slice to match validation period
        diff_moving_avg = diff_moving_avg[split_time-period-window_size:]
        
        # Add past values
        past_series = series[split_time-period:split_time]
        return past_series + diff_moving_avg
        
    return diff_moving_avg 