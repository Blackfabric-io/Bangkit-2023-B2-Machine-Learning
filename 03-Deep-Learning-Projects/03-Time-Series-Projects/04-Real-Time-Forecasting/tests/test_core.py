"""Test cases for core model functionality."""

import numpy as np
import tensorflow as tf
import pytest
from src.core.base import RealTimeModel

def test_model_initialization():
    """Test model initialization with default parameters."""
    model = RealTimeModel()
    assert model.input_width == 24
    assert model.label_width == 1
    assert model.shift == 1
    assert model.learning_rate == 0.001
    assert model.model is None

def test_model_initialization_custom():
    """Test model initialization with custom parameters."""
    model = RealTimeModel(
        input_width=48,
        label_width=12,
        shift=6,
        learning_rate=0.0005
    )
    assert model.input_width == 48
    assert model.label_width == 12
    assert model.shift == 6
    assert model.learning_rate == 0.0005

def test_model_build():
    """Test model architecture building."""
    model = RealTimeModel()
    model.build_model(num_features=3)
    
    # Check model structure
    assert isinstance(model.model, tf.keras.Model)
    assert len(model.model.layers) > 0
    
    # Check input shape
    input_shape = model.model.input_shape
    assert input_shape[1] == model.input_width
    assert input_shape[2] == 3  # num_features
    
    # Check output shape
    output_shape = model.model.output_shape
    assert output_shape[1] == model.label_width
    assert output_shape[2] == 3  # num_features

def test_model_compile():
    """Test model compilation."""
    model = RealTimeModel()
    model.build_model(num_features=2)
    model.compile_model()
    
    assert model.model.optimizer.__class__.__name__ == 'Adam'
    assert model.model.loss.__class__.__name__ == 'MeanSquaredError'

def test_model_training():
    """Test model training process."""
    # Create synthetic data
    batch_size = 32
    input_width = 24
    label_width = 1
    num_features = 2
    num_samples = 100
    
    X = np.random.randn(num_samples, input_width, num_features)
    y = np.random.randn(num_samples, label_width, num_features)
    
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    dataset = dataset.batch(batch_size)
    
    # Train model
    model = RealTimeModel(input_width=input_width, label_width=label_width)
    model.build_model(num_features=num_features)
    model.compile_model()
    
    history = model.model.fit(
        dataset,
        epochs=2,
        verbose=0
    )
    
    assert 'loss' in history.history
    assert len(history.history['loss']) == 2

def test_model_prediction():
    """Test model prediction functionality."""
    # Create synthetic data
    input_width = 24
    num_features = 2
    batch_size = 1
    
    X = np.random.randn(batch_size, input_width, num_features)
    
    # Make prediction
    model = RealTimeModel(input_width=input_width)
    model.build_model(num_features=num_features)
    model.compile_model()
    
    predictions = model.model.predict(X, verbose=0)
    
    assert predictions.shape == (batch_size, model.label_width, num_features)

def test_model_uncertainty():
    """Test model uncertainty estimation."""
    input_width = 24
    num_features = 2
    num_samples = 10
    mc_samples = 5
    
    X = np.random.randn(num_samples, input_width, num_features)
    
    model = RealTimeModel(input_width=input_width)
    model.build_model(num_features=num_features)
    model.compile_model()
    
    # Enable dropout for uncertainty estimation
    for layer in model.model.layers:
        if isinstance(layer, tf.keras.layers.Dropout):
            layer.training = True
    
    # Generate multiple predictions
    predictions = []
    for _ in range(mc_samples):
        pred = model.model.predict(X, verbose=0)
        predictions.append(pred)
    
    # Calculate mean and std
    predictions = np.array(predictions)
    mean_pred = np.mean(predictions, axis=0)
    std_pred = np.std(predictions, axis=0)
    
    assert mean_pred.shape == (num_samples, model.label_width, num_features)
    assert std_pred.shape == (num_samples, model.label_width, num_features)

def test_model_save_load(tmp_path):
    """Test model saving and loading functionality."""
    model = RealTimeModel()
    model.build_model(num_features=2)
    model.compile_model()
    
    # Save model
    save_path = tmp_path / "test_model"
    model.model.save(save_path)
    
    # Load model
    loaded_model = tf.keras.models.load_model(save_path)
    
    # Compare architectures
    assert len(model.model.layers) == len(loaded_model.layers)
    assert model.model.count_params() == loaded_model.count_params()

def test_invalid_parameters():
    """Test model initialization with invalid parameters."""
    with pytest.raises(ValueError):
        RealTimeModel(input_width=-1)
    
    with pytest.raises(ValueError):
        RealTimeModel(label_width=0)
    
    with pytest.raises(ValueError):
        RealTimeModel(shift=-5)
    
    with pytest.raises(ValueError):
        RealTimeModel(learning_rate=0) 