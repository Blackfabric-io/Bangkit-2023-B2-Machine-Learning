"""Tests for model processing functions."""

import os
import pytest
import tensorflow as tf
import numpy as np
from PIL import Image
from src.core import ResNetTransfer
from src.processors import train_model, evaluate_model, predict_image

@pytest.fixture
def sample_data_dir(tmp_path):
    """Create a temporary directory with sample image data."""
    # Create class directories
    class_dirs = ['class1', 'class2']
    for class_name in class_dirs:
        os.makedirs(tmp_path / class_name)
        
        # Create sample images
        for i in range(5):
            img = Image.fromarray(
                np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            )
            img.save(tmp_path / class_name / f'image_{i}.jpg')
    
    return str(tmp_path)

@pytest.fixture
def trained_model(sample_data_dir):
    """Create and train a model on sample data."""
    history = train_model(
        data_dir=sample_data_dir,
        num_classes=2,
        epochs=1,
        batch_size=2
    )
    
    model = ResNetTransfer(num_classes=2)
    model.load_weights(os.path.join('models', 'model_weights.h5'))
    
    return model

def test_train_model(sample_data_dir):
    """Test model training function."""
    # Train model
    history = train_model(
        data_dir=sample_data_dir,
        num_classes=2,
        epochs=1,
        batch_size=2
    )
    
    # Check history contents
    assert isinstance(history, dict)
    assert 'accuracy' in history
    assert 'loss' in history
    assert len(history['accuracy']) == 1
    
    # Check that weights were saved
    assert os.path.exists(os.path.join('models', 'model_weights.h5'))

def test_train_model_with_fine_tuning(sample_data_dir):
    """Test model training with fine-tuning."""
    history = train_model(
        data_dir=sample_data_dir,
        num_classes=2,
        epochs=1,
        fine_tune=True,
        fine_tune_epochs=1
    )
    
    # Check combined history length
    assert len(history['accuracy']) == 2  # Initial + fine-tuning

def test_train_model_invalid_data():
    """Test training with invalid data directory."""
    with pytest.raises(ValueError):
        train_model(
            data_dir="nonexistent_dir",
            num_classes=2
        )

def test_evaluate_model(trained_model, sample_data_dir):
    """Test model evaluation function."""
    # Evaluate model
    loss, accuracy = evaluate_model(
        model=trained_model,
        data_dir=sample_data_dir,
        batch_size=2
    )
    
    # Check metrics
    assert isinstance(loss, float)
    assert isinstance(accuracy, float)
    assert 0 <= accuracy <= 1

def test_evaluate_model_invalid_data(trained_model):
    """Test evaluation with invalid data directory."""
    with pytest.raises(ValueError):
        evaluate_model(
            model=trained_model,
            data_dir="nonexistent_dir"
        )

def test_predict_image(trained_model, sample_data_dir):
    """Test image prediction function."""
    # Get a sample image path
    image_path = os.path.join(sample_data_dir, 'class1', 'image_0.jpg')
    
    # Make prediction
    predicted_class, confidence = predict_image(
        model=trained_model,
        image_path=image_path,
        class_names=['class1', 'class2']
    )
    
    # Check results
    assert isinstance(predicted_class, str)
    assert predicted_class in ['class1', 'class2']
    assert isinstance(confidence, float)
    assert 0 <= confidence <= 1

def test_predict_image_invalid_input(trained_model):
    """Test prediction with invalid inputs."""
    # Invalid image path
    with pytest.raises(ValueError):
        predict_image(
            model=trained_model,
            image_path="nonexistent.jpg",
            class_names=['class1', 'class2']
        )
    
    # Empty class names
    with pytest.raises(ValueError):
        predict_image(
            model=trained_model,
            image_path="image.jpg",
            class_names=[]
        ) 