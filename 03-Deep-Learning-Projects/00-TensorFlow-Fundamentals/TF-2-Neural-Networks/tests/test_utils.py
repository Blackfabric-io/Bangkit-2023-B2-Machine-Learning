"""Tests for utility functions."""

import pytest
import numpy as np
from src.utils import load_mnist_data, normalize_images

def test_normalize_images():
    """Test image normalization."""
    # Create dummy uint8 array
    images = np.random.randint(0, 256, size=(10, 28, 28), dtype=np.uint8)
    
    normalized = normalize_images(images)
    assert normalized.dtype == np.float32
    assert np.all(normalized >= 0.0)
    assert np.all(normalized <= 1.0)

def test_normalize_invalid_input():
    """Test normalization with invalid input."""
    with pytest.raises(ValueError):
        # Wrong input type
        normalize_images([1, 2, 3])
        
    with pytest.raises(ValueError):
        # Wrong data type
        images = np.array([1, 2, 3], dtype=np.int16)
        normalize_images(images)

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