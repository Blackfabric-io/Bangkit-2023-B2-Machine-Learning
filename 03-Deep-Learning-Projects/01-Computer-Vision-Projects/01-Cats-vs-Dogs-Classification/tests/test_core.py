"""Tests for core model and callback functionality."""

import pytest
import numpy as np
import tensorflow as tf
from src.core import CatDogModel, AccuracyCallback

def test_model_initialization():
    """Test that model can be initialized with correct architecture."""
    model = CatDogModel()
    
    # Check model structure
    assert isinstance(model.model, tf.keras.Model)
    assert len(model.model.layers) == 15  # 5 conv, 5 pool, flatten, 2 dense
    
    # Check input shape
    assert model.model.input_shape == (None, 150, 150, 3)
    
    # Check output shape
    assert model.model.output_shape == (None, 1)

def test_callback_initialization():
    """Test that callback can be initialized with custom threshold."""
    # Default threshold
    callback = AccuracyCallback()
    assert callback.threshold == 0.99
    
    # Custom threshold
    callback = AccuracyCallback(threshold=0.95)
    assert callback.threshold == 0.95

def test_model_prediction():
    """Test that model can make predictions with valid input."""
    model = CatDogModel()
    
    # Create dummy input
    image = np.random.random((150, 150, 3))
    image = np.expand_dims(image, axis=0)
    
    # Get prediction
    pred = model.predict(image)
    
    # Check prediction shape and range
    assert isinstance(pred, np.ndarray)
    assert pred.shape == (1, 1)
    assert np.all(pred >= 0) and np.all(pred <= 1)

def test_invalid_input_shape():
    """Test that model raises error for invalid input shapes."""
    model = CatDogModel()
    
    # Wrong spatial dimensions
    with pytest.raises(ValueError):
        image = np.random.random((1, 100, 100, 3))
        model.predict(image)
    
    # Wrong number of channels
    with pytest.raises(ValueError):
        image = np.random.random((1, 150, 150, 1))
        model.predict(image)
    
    # Wrong number of dimensions
    with pytest.raises(ValueError):
        image = np.random.random((150, 150, 3))
        model.predict(image)

def test_callback_stops_training():
    """Test that callback stops training when accuracy threshold is met."""
    model = CatDogModel()
    callback = AccuracyCallback(threshold=0.8)
    
    # Create dummy data
    x = np.random.random((100, 150, 150, 3))
    y = np.random.randint(0, 2, (100,))
    
    # Create dummy generator
    class DummyGenerator:
        def __init__(self):
            self.x = x
            self.y = y
            self.__class__ = tf.keras.preprocessing.image.DirectoryIterator
            
        def __iter__(self):
            return self
            
        def __next__(self):
            return self.x, self.y
    
    # Train model with callback
    history = model.train(DummyGenerator(), epochs=10, callback_threshold=0.8, verbose=0)
    assert len(history.history['accuracy']) < 10  # Training should stop early

def test_model_save_load(tmp_path):
    """Test that model weights can be saved and loaded."""
    model = CatDogModel()
    
    # Create temporary file path
    weights_path = tmp_path / "model_weights.h5"
    
    # Save weights
    model.save(str(weights_path))
    assert weights_path.exists()
    
    # Create new model and load weights
    new_model = CatDogModel()
    new_model.load(str(weights_path))
    
    # Compare predictions
    test_input = np.random.random((1, 150, 150, 3))
    pred1 = model.predict(test_input)
    pred2 = new_model.predict(test_input)
    
    assert np.allclose(pred1, pred2) 