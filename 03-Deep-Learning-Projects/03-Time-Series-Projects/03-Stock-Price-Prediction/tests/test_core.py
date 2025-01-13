"""Test cases for core model functionality."""

import numpy as np
import tensorflow as tf
import pytest
from src.core.base import RNNModel, LSTMModel, BiLSTMModel

def test_model_initialization():
    """Test model initialization with default parameters."""
    model = RNNModel()
    
    assert model.sequence_length == 60
    assert model.n_features == 1
    assert model.batch_size == 32
    assert model.model is None

def test_model_building():
    """Test model architecture building."""
    models = [RNNModel(), LSTMModel(), BiLSTMModel()]
    
    for model in models:
        model.build_model()
        
        assert model.model is not None
        assert isinstance(model.model, tf.keras.Sequential)
        
        # Check input shape handling
        sample_input = np.random.randn(1, 60, 1).astype(np.float32)
        output = model.model(sample_input)
        assert output.shape == (1, 1)  # (batch, prediction)

def test_model_compilation():
    """Test model compilation."""
    model = LSTMModel()
    model.compile_model(learning_rate=0.01)
    
    assert model.model is not None
    assert isinstance(model.model.optimizer, tf.keras.optimizers.Adam)
    assert model.model.optimizer.learning_rate == 0.01

def test_model_training():
    """Test model training workflow."""
    # Create synthetic data
    X_train = np.random.randn(100, 60, 1).astype(np.float32)
    y_train = np.random.randn(100, 1).astype(np.float32)
    X_val = np.random.randn(20, 60, 1).astype(np.float32)
    y_val = np.random.randn(20, 1).astype(np.float32)
    
    model = RNNModel()
    history = model.train(
        X_train, y_train,
        X_val, y_val,
        epochs=2
    )
    
    assert 'loss' in history
    assert 'val_loss' in history
    assert len(history['loss']) <= 2

def test_model_prediction():
    """Test model prediction functionality."""
    model = LSTMModel()
    
    # Should raise error if not trained
    with pytest.raises(ValueError):
        model.predict(np.random.randn(10, 60, 1))
    
    # Build and compile model
    model.compile_model()
    
    # Test prediction shape
    test_input = np.random.randn(5, 60, 1).astype(np.float32)
    predictions = model.predict(test_input)
    
    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == (5, 1)  # (samples, prediction)

def test_model_inheritance():
    """Test model inheritance structure."""
    models = [RNNModel(), LSTMModel(), BiLSTMModel()]
    
    for model in models:
        # Test method inheritance
        assert hasattr(model, 'train')
        assert hasattr(model, 'predict')
        assert hasattr(model, 'compile_model')
        
        # Test attribute inheritance
        assert hasattr(model, 'sequence_length')
        assert hasattr(model, 'n_features')
        assert hasattr(model, 'batch_size') 