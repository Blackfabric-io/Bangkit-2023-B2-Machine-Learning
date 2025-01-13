import pytest
import tensorflow as tf
import numpy as np
from src.core import RockPaperScissorsModel, format_example

def test_format_example():
    # Create dummy image and label
    feature = tf.ones((300, 300, 3), dtype=tf.uint8)
    label = tf.constant(0)
    
    # Format the example
    formatted_feature, formatted_label = format_example(feature, label)
    
    # Check feature remains unchanged
    assert formatted_feature.shape == (300, 300, 3)
    
    # Check one-hot encoding
    assert formatted_label.shape == (3,)
    assert tf.reduce_sum(formatted_label).numpy() == 1.0
    assert formatted_label[0] == 1.0

def test_model_initialization():
    model = RockPaperScissorsModel(batch_size=32)
    assert model.batch_size == 32
    assert model.model is None
    assert len(model.class_names) == 3

def test_model_build():
    model = RockPaperScissorsModel()
    model.build_model()
    
    # Check model structure
    assert isinstance(model.model, tf.keras.Sequential)
    assert len(model.model.layers) == 4
    
    # Check input shape
    assert model.model.input_shape == (None, 300, 300, 3)
    
    # Check output shape
    assert model.model.output_shape == (None, 3)

def test_model_train_validation():
    model = RockPaperScissorsModel()
    
    # Try to train without building
    with pytest.raises(ValueError):
        model.train(None, None)
    
    # Try to get model info without building
    with pytest.raises(ValueError):
        model.get_model_info() 