"""Test cases for core functionality."""

import numpy as np
import pytest
from src.core.base import (trend, seasonal_pattern, seasonality,
                          noise, train_val_split, compute_metrics)

def test_trend():
    """Test trend generation."""
    time = np.array([0, 1, 2], dtype=np.float32)
    slope = 0.5
    
    result = trend(time, slope)
    expected = np.array([0, 0.5, 1.0], dtype=np.float32)
    
    np.testing.assert_array_almost_equal(result, expected)

def test_seasonal_pattern():
    """Test seasonal pattern generation."""
    season_time = np.array([0, 0.05, 0.5], dtype=np.float32)
    
    result = seasonal_pattern(season_time)
    # First two points should be cosine values, last one exponential decay
    expected = np.array([1.0, 0.0, 0.082], dtype=np.float32)
    
    np.testing.assert_array_almost_equal(result, expected, decimal=3)

def test_seasonality():
    """Test seasonality generation."""
    time = np.array([0, 50, 100], dtype=np.float32)
    period = 100
    amplitude = 2
    
    result = seasonality(time, period, amplitude)
    # Should repeat pattern every period
    assert np.isclose(result[0], result[-1])
    assert len(result) == len(time)

def test_noise():
    """Test noise generation."""
    time = np.array([0, 1, 2], dtype=np.float32)
    noise_level = 1.0
    seed = 42
    
    result = noise(time, noise_level, seed)
    
    assert len(result) == len(time)
    assert abs(result.mean()) < 0.5  # Should be roughly centered at 0
    assert abs(result.std() - noise_level) < 0.5  # Should have specified std

def test_train_val_split():
    """Test time series splitting."""
    time = np.arange(10, dtype=np.float32)
    series = np.arange(10, dtype=np.float32)
    split_time = 7
    
    time_train, series_train, time_valid, series_valid = train_val_split(
        time, series, split_time)
    
    assert len(time_train) == split_time
    assert len(series_train) == split_time
    assert len(time_valid) == len(time) - split_time
    assert len(series_valid) == len(series) - split_time
    
    np.testing.assert_array_equal(time_train, time[:split_time])
    np.testing.assert_array_equal(series_train, series[:split_time])
    np.testing.assert_array_equal(time_valid, time[split_time:])
    np.testing.assert_array_equal(series_valid, series[split_time:])

def test_compute_metrics():
    """Test evaluation metrics computation."""
    true_series = np.array([1, 2, 3], dtype=np.float32)
    forecast = np.array([2, 2, 4], dtype=np.float32)
    
    mse, mae = compute_metrics(true_series, forecast)
    
    assert mse == pytest.approx(1.0)  # (1² + 0² + 1²) / 3
    assert mae == pytest.approx(2/3)  # (1 + 0 + 1) / 3 