import pytest
import tensorflow as tf
import numpy as np
from src.core import FashionMNISTModel, format_example

def test_format_example():
    # Create dummy image and label
    image = tf.ones((28, 28, 1), dtype=tf.uint8) * 255
    label = tf.constant(0)
    
    # Format the example
    formatted_image, formatted_label = format_example(image, label)
    
    # Check normalization
    assert formatted_image.dtype == tf.float32
    assert tf.reduce_max(formatted_image).numpy() <= 1.0
    assert tf.reduce_min(formatted_image).numpy() >= 0.0
    
    # Check label remains unchanged
    assert formatted_label == label

def test_model_initialization():
    model = FashionMNISTModel(batch_size=32)
    assert model.batch_size == 32
    assert model.model is None
    assert len(model.class_names) == 10

def test_model_build():
    model = FashionMNISTModel()
    model.build_model()
    
    # Check model structure
    assert isinstance(model.model, tf.keras.Sequential)
    assert len(model.model.layers) == 6
    
    # Check input shape
    assert model.model.input_shape == (None, 28, 28, 1)
    
    # Check output shape
    assert model.model.output_shape == (None, 10)

def test_model_train_validation():
    model = FashionMNISTModel()
    
    # Try to train without building
    with pytest.raises(ValueError):
        model.train(None, None)
    
    # Try to convert without training
    with pytest.raises(ValueError):
        model.convert_to_tflite("test_dir") 