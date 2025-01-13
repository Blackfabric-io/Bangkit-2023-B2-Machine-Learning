"""Tests for the core model and callback functionality."""

import pytest
import numpy as np
import tensorflow as tf
from src.core import CNNModel, AccuracyThresholdCallback

def test_model_initialization():
    """Test that model can be initialized."""
    model = CNNModel()
    assert model is not None
    assert model.model is not None
    assert isinstance(model.model, tf.keras.Model)

def test_callback_initialization():
    """Test that callback can be initialized with custom threshold."""
    threshold = 0.995
    callback = AccuracyThresholdCallback(threshold=threshold)
    assert callback is not None
    assert callback.threshold == threshold

def test_model_training():
    """Test that model can be trained on dummy data."""
    model = CNNModel()
    
    # Create dummy data
    x_train = np.random.rand(100, 28, 28, 1)
    y_train = np.random.randint(0, 10, size=100)
    
    history = model.train(x_train, y_train, epochs=1, verbose=0)
    assert history is not None
    assert isinstance(history, tf.keras.callbacks.History)
    assert 'accuracy' in history.history

def test_model_prediction():
    """Test that model can make predictions."""
    model = CNNModel()
    
    # Create dummy input
    x = np.random.rand(1, 28, 28, 1)
    
    prediction = model.predict(x)
    assert prediction is not None
    assert isinstance(prediction, np.ndarray)
    assert prediction.shape == (1, 10)
    np.testing.assert_almost_equal(np.sum(prediction[0]), 1.0)

def test_invalid_input_shape():
    """Test that model raises error for invalid input shape."""
    model = CNNModel()
    
    with pytest.raises(ValueError):
        # Wrong input shape
        x = np.random.rand(1, 32, 32, 1)
        model.predict(x)
        
    with pytest.raises(ValueError):
        # Missing channel dimension
        x = np.random.rand(1, 28, 28)
        model.predict(x)

def test_callback_stops_training():
    """Test that callback stops training at threshold."""
    model = CNNModel()
    callback = AccuracyThresholdCallback(threshold=0.0)  # Set threshold to 0 to ensure stopping
    
    # Create dummy data
    x_train = np.random.rand(100, 28, 28, 1)
    y_train = np.random.randint(0, 10, size=100)
    
    history = model.train(x_train, y_train, epochs=10, callback_threshold=0.0, verbose=0)
    assert len(history.history['accuracy']) < 10  # Training should stop early 