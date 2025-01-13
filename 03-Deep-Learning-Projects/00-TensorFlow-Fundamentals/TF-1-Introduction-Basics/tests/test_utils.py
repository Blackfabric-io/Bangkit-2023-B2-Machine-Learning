"""Tests for utility functions."""

import pytest
import numpy as np
from src.utils import scale_price, unscale_price

def test_scale_price_single():
    """Test scaling single price value."""
    price = 150.0  # 150k
    scaled = scale_price(price)
    assert scaled[0] == 1.5

def test_scale_price_array():
    """Test scaling array of prices."""
    prices = [100.0, 150.0, 200.0]
    scaled = scale_price(prices)
    np.testing.assert_array_equal(scaled, np.array([1.0, 1.5, 2.0]))

def test_unscale_price_single():
    """Test unscaling single price value."""
    scaled_price = 1.5  # 150k
    unscaled = unscale_price(scaled_price)
    assert unscaled[0] == 150.0

def test_unscale_price_array():
    """Test unscaling array of prices."""
    scaled_prices = [1.0, 1.5, 2.0]
    unscaled = unscale_price(scaled_prices)
    np.testing.assert_array_equal(unscaled, np.array([100.0, 150.0, 200.0])) 