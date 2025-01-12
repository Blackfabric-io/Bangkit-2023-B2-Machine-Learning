"""
Unit tests for linear regression implementation.
"""

import numpy as np
import pytest
from src.core.base import LinearRegression

def test_model_initialization():
    """Test model initialization with default and custom parameters."""
    # Test default parameters
    model = LinearRegression()
    assert model.learning_rate == 0.01
    assert model.n_iterations == 1500
    assert model.w is None
    assert model.b is None
    assert model.cost_history == []
    
    # Test custom parameters
    model = LinearRegression(learning_rate=0.1, n_iterations=1000)
    assert model.learning_rate == 0.1
    assert model.n_iterations == 1000

def test_compute_cost():
    """Test cost computation."""
    model = LinearRegression()
    model.w = 2.0
    model.b = 1.0
    
    X = np.array([1.0, 2.0, 3.0])
    y = np.array([4.0, 7.0, 10.0])
    
    cost = model.compute_cost(X, y)
    assert isinstance(cost, float)
    assert cost >= 0

def test_compute_gradient():
    """Test gradient computation."""
    model = LinearRegression()
    model.w = 2.0
    model.b = 1.0
    
    X = np.array([1.0, 2.0, 3.0])
    y = np.array([4.0, 7.0, 10.0])
    
    dj_dw, dj_db = model.compute_gradient(X, y)
    assert isinstance(dj_dw, float)
    assert isinstance(dj_db, float)

def test_fit_predict():
    """Test model fitting and prediction."""
    # Generate synthetic data
    np.random.seed(42)
    X = np.random.randn(100)
    true_w, true_b = 2.5, -1.0
    y = true_w * X + true_b + np.random.randn(100) * 0.1
    
    # Fit model
    model = LinearRegression(learning_rate=0.1, n_iterations=1000)
    model.fit(X, y)
    
    # Check if parameters are learned
    assert model.w is not None
    assert model.b is not None
    assert len(model.cost_history) > 0
    
    # Check predictions
    y_pred = model.predict(X)
    assert y_pred.shape == y.shape
    
    # Check if learned parameters are close to true parameters
    assert abs(model.w - true_w) < 0.5
    assert abs(model.b - true_b) < 0.5

def test_input_validation():
    """Test input validation and error handling."""
    model = LinearRegression()
    
    # Test uninitialized model
    X = np.array([1.0, 2.0])
    with pytest.raises(ValueError):
        model.predict(X)
    
    with pytest.raises(ValueError):
        model.compute_cost(X, X)
    
    with pytest.raises(ValueError):
        model.compute_gradient(X, X)
    
    # Test mismatched dimensions
    X = np.array([1.0, 2.0])
    y = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        model.fit(X, y) 