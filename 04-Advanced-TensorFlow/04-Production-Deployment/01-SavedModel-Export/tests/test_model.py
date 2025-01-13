"""
Tests for the MNIST model class.
"""

import os
import tempfile
import pytest
import tensorflow as tf
import numpy as np
from src.core.model import MNIST

@pytest.fixture
def model():
    """Fixture to create a temporary MNIST model instance."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        model = MNIST(
            export_path=tmp_dir,
            buffer_size=100,  # Small buffer for testing
            batch_size=32,
            learning_rate=1e-3,
            epochs=1  # Single epoch for testing
        )
        yield model

def test_model_initialization(model):
    """Test model initialization."""
    assert model._buffer_size == 100
    assert model._batch_size == 32
    assert model._learning_rate == 1e-3
    assert model._epochs == 1
    assert isinstance(model._model, tf.keras.Model)

def test_preprocess_fn(model):
    """Test image preprocessing function."""
    # Create dummy input
    x = tf.ones((28, 28, 1), dtype=tf.uint8) * 255
    
    # Apply preprocessing
    result = model.preprocess_fn(x)
    
    # Check output
    assert result.dtype == tf.float32
    assert tf.reduce_max(result) <= 1.0
    assert tf.reduce_min(result) >= 0.0

def test_model_predict(model):
    """Test model prediction."""
    # Create dummy batch
    batch = tf.ones((1, 28, 28, 1), dtype=tf.uint8)
    
    # Get predictions
    predictions = model.predict(batch)
    
    # Check predictions shape and values
    assert predictions.shape == (1, 10)
    assert np.allclose(np.sum(predictions), 1.0)
    assert np.all(predictions >= 0.0) and np.all(predictions <= 1.0)

def test_model_export(model):
    """Test model export."""
    # Export model
    model.export_model()
    
    # Check if export directory exists
    assert os.path.exists(model._export_path)
    
    # Check if SavedModel files exist
    assert os.path.exists(os.path.join(model._export_path, 'saved_model.pb'))
    assert os.path.exists(os.path.join(model._export_path, 'variables'))

def test_dataset_preparation(model):
    """Test dataset preparation."""
    # Check if datasets are TF datasets
    assert isinstance(model.train_dataset, tf.data.Dataset)
    assert isinstance(model.test_dataset, tf.data.Dataset)
    
    # Check if datasets return correct types
    for images, labels in model.train_dataset.take(1):
        assert images.dtype == tf.uint8
        assert labels.dtype == tf.int64
        assert images.shape[-1] == 1  # Single channel
        break 