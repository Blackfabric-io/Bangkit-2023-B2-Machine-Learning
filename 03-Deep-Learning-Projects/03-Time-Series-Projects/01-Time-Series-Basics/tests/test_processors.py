"""Test cases for forecasting processors."""

import numpy as np
import pytest
from src.processors.main import (moving_average_forecast,
                               naive_forecast,
                               diff_forecast)

def test_moving_average_forecast():
    """Test moving average forecasting."""
    series = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    window_size = 3
    
    result = moving_average_forecast(series, window_size)
    expected = np.array([2, 3, 4], dtype=np.float32)  # [mean(1,2,3), mean(2,3,4), mean(3,4,5)]
    
    np.testing.assert_array_almost_equal(result, expected)

def test_naive_forecast():
    """Test naive forecasting."""
    series = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    split_time = 3
    
    result = naive_forecast(series, split_time)
    expected = np.array([3, 4], dtype=np.float32)  # Uses previous values
    
    np.testing.assert_array_equal(result, expected)

def test_diff_forecast():
    """Test differenced moving average forecasting."""
    # Create synthetic series with clear seasonality
    period = 3
    series = np.array([1, 2, 3, 2, 3, 4, 3, 4, 5], dtype=np.float32)
    window_size = 2
    split_time = 6
    
    result = diff_forecast(series, period, window_size, split_time)
    
    # Check basic properties
    assert len(result) == len(series) - split_time
    # Differences should be similar across periods
    diffs = np.diff(series)
    assert np.allclose(diffs[:period], diffs[period:2*period], atol=0.1)

def test_diff_forecast_no_split():
    """Test differenced forecast without splitting."""
    series = np.array([1, 2, 3, 2, 3, 4], dtype=np.float32)
    period = 3
    window_size = 2
    
    result = diff_forecast(series, period, window_size)
    
    # Should return full differenced series
    assert len(result) == len(series) - period - window_size + 1 