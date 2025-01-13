"""Tests for core sign language recognition functionality."""

import pytest
import tensorflow as tf
import numpy as np
from src.core import SignLanguageModel, TrainingCallback

@pytest.fixture
def sample_model():
    """Create a sample model for testing."""
    return SignLanguageModel(num_classes=26)  # A-Z

@pytest.fixture
def sample_data():
    """Create sample training data."""
    # Create random images and labels
    images = np.random.rand(10, 64, 64, 1).astype(np.float32)
    labels = np.eye(26)[np.random.randint(0, 26, 10)]
    
    # Create tf.data.Dataset
    return tf.data.Dataset.from_tensor_slices((images, labels)).batch(2)

def test_model_initialization():
    """Test model initialization with different parameters."""
    # Default parameters
    model = SignLanguageModel(num_classes=26)
    assert model.num_classes == 26
    assert model.input_shape == (64, 64, 1)
    
    # Custom parameters
    model = SignLanguageModel(
        num_classes=10,
        input_shape=(32, 32, 1),
        learning_rate=0.0001
    )
    assert model.num_classes == 10
    assert model.input_shape == (32, 32, 1)
    assert model.learning_rate == 0.0001

def test_model_invalid_params():
    """Test model initialization with invalid parameters."""
    # Invalid number of classes
    with pytest.raises(ValueError):
        SignLanguageModel(num_classes=1)
    
    # Invalid input shape
    with pytest.raises(ValueError):
        SignLanguageModel(num_classes=26, input_shape=(0, 64, 1))
    
    # Invalid learning rate
    with pytest.raises(ValueError):
        SignLanguageModel(num_classes=26, learning_rate=0)

def test_model_output_shape(sample_model):
    """Test model output shape."""
    # Create dummy input
    dummy_input = tf.random.uniform((1, 64, 64, 1))
    
    # Get prediction
    output = sample_model.model(dummy_input)
    
    # Check shape
    assert output.shape == (1, 26)  # (batch_size, num_classes)
    
    # Check value range
    assert tf.reduce_all(output >= 0)
    assert tf.reduce_all(output <= 1)
    assert np.allclose(tf.reduce_sum(output, axis=1), 1)

def test_model_training(sample_model, sample_data):
    """Test model training functionality."""
    # Train for a few steps
    history = sample_model.train(
        sample_data,
        epochs=2
    )
    
    # Check history contents
    assert 'accuracy' in history
    assert 'loss' in history
    assert len(history['accuracy']) == 2
    assert len(history['loss']) == 2

def test_model_prediction(sample_model):
    """Test model prediction functionality."""
    # Create test image
    image = np.random.rand(64, 64, 1).astype(np.float32)
    
    # Test class prediction
    pred_class = sample_model.predict(image)
    assert isinstance(pred_class, int)
    assert 0 <= pred_class < 26
    
    # Test probability prediction
    probs = sample_model.predict(image, return_probabilities=True)
    assert isinstance(probs, np.ndarray)
    assert probs.shape == (26,)
    assert np.allclose(np.sum(probs), 1)

def test_model_evaluation(sample_model, sample_data):
    """Test model evaluation functionality."""
    # Evaluate model
    loss, accuracy = sample_model.evaluate(sample_data)
    
    # Check metrics
    assert isinstance(loss, float)
    assert isinstance(accuracy, float)
    assert 0 <= accuracy <= 1

def test_callback_initialization():
    """Test training callback initialization."""
    # Default parameters
    callback = TrainingCallback()
    assert callback.accuracy_threshold == 0.95
    assert callback.patience == 3
    
    # Custom parameters
    callback = TrainingCallback(accuracy_threshold=0.9, patience=5)
    assert callback.accuracy_threshold == 0.9
    assert callback.patience == 5

def test_callback_monitoring():
    """Test callback monitoring functionality."""
    callback = TrainingCallback(accuracy_threshold=0.8, patience=2)
    
    # Mock model and logs
    class MockModel:
        def __init__(self):
            self.stop_training = False
    
    callback.model = MockModel()
    
    # Test accuracy threshold
    callback.on_epoch_end(0, {'accuracy': 0.9})
    assert callback.model.stop_training  # Should stop when threshold reached
    
    # Test patience
    callback.model.stop_training = False
    callback.wait = 0
    callback.best_acc = 0
    
    callback.on_epoch_end(0, {'accuracy': 0.5})
    assert not callback.model.stop_training
    
    callback.on_epoch_end(1, {'accuracy': 0.4})
    assert not callback.model.stop_training
    
    callback.on_epoch_end(2, {'accuracy': 0.4})
    assert callback.model.stop_training  # Should stop after patience exceeded

def test_model_weight_saving(sample_model, tmp_path):
    """Test model weight saving and loading."""
    # Save weights
    weights_path = tmp_path / "weights.h5"
    sample_model.save_weights(str(weights_path))
    assert weights_path.exists()
    
    # Load weights
    new_model = SignLanguageModel(num_classes=26)
    new_model.load_weights(str(weights_path))
    
    # Compare model outputs
    test_input = tf.random.uniform((1, 64, 64, 1))
    assert np.allclose(
        sample_model.model(test_input),
        new_model.model(test_input)
    ) 