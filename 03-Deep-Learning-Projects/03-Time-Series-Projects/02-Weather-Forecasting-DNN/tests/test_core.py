"""Test cases for core DNN model functionality."""

import numpy as np
import tensorflow as tf
import pytest
from src.core.base import WeatherModel

def test_model_initialization():
    """Test model initialization with default parameters."""
    model = WeatherModel()
    
    assert model.input_width == 5
    assert model.label_width == 1
    assert model.shift == 1
    assert model.batch_size == 32
    assert model.total_window_size == 6
    assert model.model is None

def test_split_window():
    """Test window splitting functionality."""
    model = WeatherModel(input_width=3, label_width=2, shift=2)
    
    # Create sample data
    data = np.array([
        [1, 2],  # features
        [3, 4],
        [5, 6],
        [7, 8],
        [9, 10]
    ], dtype=np.float32)
    
    # Add batch dimension
    data = tf.expand_dims(data, axis=0)
    
    inputs, labels = model.split_window(data)
    
    assert inputs.shape == (1, 3, 2)  # (batch, input_width, features)
    assert labels.shape == (1, 2, 2)  # (batch, label_width, features)

def test_make_dataset():
    """Test dataset creation."""
    model = WeatherModel(input_width=2, label_width=1, shift=1, batch_size=2)
    
    # Create sample data
    data = np.array([
        [1, 2],
        [3, 4],
        [5, 6],
        [7, 8]
    ], dtype=np.float32)
    
    ds = model.make_dataset(data, shuffle=False)
    
    # Check dataset structure
    for inputs, labels in ds.take(1):
        assert inputs.shape == (2, 2, 2)  # (batch, input_width, features)
        assert labels.shape == (2, 1, 2)  # (batch, label_width, features)

def test_model_building():
    """Test model architecture building."""
    model = WeatherModel()
    model.build_model(num_features=3)
    
    assert model.model is not None
    assert len(model.model.layers) == 3
    
    # Check input shape handling
    sample_input = np.random.randn(1, 5, 3).astype(np.float32)
    output = model.model(sample_input)
    assert output.shape == (1, 3)  # (batch, features)

def test_prediction():
    """Test model prediction functionality."""
    model = WeatherModel(input_width=2, label_width=1, shift=1)
    
    # Should raise error if not trained
    with pytest.raises(ValueError):
        model.predict(np.random.randn(10, 2))
    
    # Build and compile model
    model.build_model(num_features=2)
    
    # Test prediction shape
    test_input = np.random.randn(4, 2).astype(np.float32)
    predictions = model.predict(test_input)
    
    assert isinstance(predictions, np.ndarray)
    assert predictions.shape[1] == 2  # num_features 