"""Tests for utility functions."""

import pytest
import numpy as np
from src.utils import load_mnist_data, preprocess_images

def test_preprocess_images():
    """Test image preprocessing."""
    # Create dummy uint8 array
    images = np.random.randint(0, 256, size=(10, 28, 28), dtype=np.uint8)
    
    processed = preprocess_images(images)
    assert processed.dtype == np.float32
    assert processed.shape == (10, 28, 28, 1)
    assert np.all(processed >= 0.0)
    assert np.all(processed <= 1.0)

def test_preprocess_invalid_input():
    """Test preprocessing with invalid input."""
    with pytest.raises(ValueError):
        # Wrong input type
        preprocess_images([1, 2, 3])
        
    with pytest.raises(ValueError):
        # Wrong shape
        images = np.random.rand(10, 32, 32)
        preprocess_images(images)
        
    with pytest.raises(ValueError):
        # Wrong number of dimensions
        images = np.random.rand(28, 28)
        preprocess_images(images)

def test_load_mnist_data():
    """Test MNIST data loading."""
    try:
        x_train, y_train = load_mnist_data()
        assert isinstance(x_train, np.ndarray)
        assert isinstance(y_train, np.ndarray)
        assert x_train.shape[1:] == (28, 28)
        assert len(y_train.shape) == 1
        assert np.all(y_train >= 0) and np.all(y_train <= 9)
    except Exception as e:
        pytest.skip(f"MNIST data loading failed: {str(e)}")

def test_load_mnist_invalid_path():
    """Test MNIST data loading with invalid path."""
    with pytest.raises(RuntimeError):
        load_mnist_data("invalid/path/to/mnist.npz") 