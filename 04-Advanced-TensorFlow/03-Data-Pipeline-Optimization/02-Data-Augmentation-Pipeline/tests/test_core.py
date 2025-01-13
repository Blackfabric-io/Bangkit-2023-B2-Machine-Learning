import pytest
import tensorflow as tf
import numpy as np
from src.core import CatDogClassifier, format_image

def test_format_image():
    # Create dummy image and label
    features = {
        'image': tf.ones((300, 300, 3), dtype=tf.uint8),
        'label': tf.constant(0)
    }
    
    # Format the example
    formatted_image, label = format_image(features)
    
    # Check image shape and normalization
    assert formatted_image.shape == (224, 224, 3)
    assert tf.reduce_max(formatted_image).numpy() <= 1.0
    assert tf.reduce_min(formatted_image).numpy() >= 0.0
    
    # Check label remains unchanged
    assert label == features['label']

def test_model_initialization():
    model = CatDogClassifier(batch_size=32, num_examples=100)
    assert model.batch_size == 32
    assert model.num_examples == 100
    assert model.model is None
    assert model.feature_extractor is None

def test_model_build_without_feature_extractor():
    model = CatDogClassifier()
    
    # Try to build without loading feature extractor
    with pytest.raises(ValueError):
        model.build_model()

def test_model_train_without_building():
    model = CatDogClassifier()
    
    # Try to train without building
    with pytest.raises(ValueError):
        model.train(None, None)
    
    # Try to evaluate without building
    with pytest.raises(ValueError):
        model.evaluate(None)
    
    # Try to get model info without building
    with pytest.raises(ValueError):
        model.get_model_info() 