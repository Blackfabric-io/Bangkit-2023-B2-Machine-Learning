"""Tests for core model and callback functionality."""

import pytest
import numpy as np
import tensorflow as tf
from src.core.base import EmotionModel, AccuracyThresholdCallback

def test_model_initialization():
    """Test that model can be initialized with correct architecture."""
    model = EmotionModel()
    
    # Check model structure
    assert isinstance(model.model, tf.keras.Model)
    assert len(model.model.layers) == 9  # 3 conv, 3 pool, flatten, 2 dense
    
    # Check input shape
    assert model.model.input_shape == (None, 28, 28, 1)
    
    # Check output shape
    assert model.model.output_shape == (None, 1)

def test_callback_initialization():
    """Test that callback can be initialized with custom threshold."""
    # Default threshold
    callback = AccuracyThresholdCallback()
    assert callback.threshold == 0.999
    
    # Custom threshold
    callback = AccuracyThresholdCallback(threshold=0.95)
    assert callback.threshold == 0.95

def test_model_prediction():
    """Test that model can make predictions with valid input."""
    model = EmotionModel()
    
    # Create dummy input
    image = np.random.random((28, 28, 1))
    
    # Get prediction
    pred = model.predict(image)
    
    # Check prediction shape and range
    assert isinstance(pred, float)
    assert 0 <= pred <= 1

def test_invalid_input_shape():
    """Test that model raises error for invalid input shapes."""
    model = EmotionModel()
    
    # Wrong number of channels
    with pytest.raises(ValueError):
        image = np.random.random((28, 28, 3))
        model.predict(image)
    
    # Wrong spatial dimensions
    with pytest.raises(ValueError):
        image = np.random.random((32, 32, 1))
        model.predict(image)
    
    # Wrong number of dimensions
    with pytest.raises(ValueError):
        image = np.random.random((28, 28))
        model.predict(image)

def test_callback_stops_training():
    """Test that callback stops training when accuracy threshold is met."""
    model = EmotionModel()
    callback = AccuracyThresholdCallback(threshold=0.8)
    
    # Create dummy data
    x = np.random.random((100, 28, 28, 1))
    y = np.random.randint(0, 2, (100,))
    
    # Train model with callback
    history = model.model.fit(
        x, y,
        epochs=10,
        callbacks=[callback],
        verbose=0
    )
    
    # Check if training stopped early
    assert len(history.history['accuracy']) < 10

def test_model_save_load(tmp_path):
    """Test that model weights can be saved and loaded."""
    model = EmotionModel()
    
    # Create temporary file path
    weights_path = tmp_path / "model_weights.h5"
    
    # Save weights
    model.save(weights_path)
    assert weights_path.exists()
    
    # Create new model and load weights
    new_model = EmotionModel()
    new_model.load(weights_path)
    
    # Compare predictions
    test_input = np.random.random((28, 28, 1))
    pred1 = model.predict(test_input)
    pred2 = new_model.predict(test_input)
    
    assert np.allclose(pred1, pred2)

def test_model_training():
    """Test that model can be trained on dummy data."""
    model = EmotionModel()
    
    # Create dummy training data
    x = np.random.random((100, 28, 28, 1))
    y = np.random.randint(0, 2, (100,))
    
    # Convert to tf.data.Dataset
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.batch(10)
    
    # Train for one epoch
    history = model.train(dataset, epochs=1)
    
    # Check that history contains expected metrics
    assert 'loss' in history.history
    assert 'accuracy' in history.history 