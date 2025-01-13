"""Tests for text processing functionality."""

import pytest
import numpy as np
import tensorflow as tf
from src.processors import process_data, train_model, evaluate_model

@pytest.fixture
def sample_data_file(tmp_path):
    """Create sample data file."""
    data_file = tmp_path / "test.csv"
    data_file.write_text(
        "category,text\n"
        "tech,This is a technology article about computers\n"
        "sport,Sports news about football and tennis\n"
        "business,Economic news about stock markets"
    )
    return str(data_file)

@pytest.fixture
def processed_data(sample_data_file):
    """Process sample data."""
    data, tokenizer = process_data(sample_data_file)
    return data, tokenizer

def test_process_data(sample_data_file):
    """Test data processing."""
    # Process data
    data, tokenizer = process_data(sample_data_file)
    
    # Check data structure
    assert 'train_sequences' in data
    assert 'train_labels' in data
    
    # Check sequences
    assert isinstance(data['train_sequences'], np.ndarray)
    assert len(data['train_sequences']) == 3
    
    # Check labels
    assert isinstance(data['train_labels'], list)
    assert len(data['train_labels']) == 3

def test_train_model(processed_data):
    """Test model training."""
    data, tokenizer = processed_data
    
    # Train model
    model, history = train_model(
        train_sequences=data['train_sequences'],
        train_labels=data['train_labels'],
        num_classes=3,
        epochs=2
    )
    
    # Check model
    assert isinstance(model, tf.keras.Model)
    
    # Check history
    assert 'accuracy' in history
    assert 'loss' in history
    assert len(history['accuracy']) == 2
    assert len(history['loss']) == 2

def test_evaluate_model(processed_data):
    """Test model evaluation."""
    data, tokenizer = processed_data
    
    # Train model first
    model, _ = train_model(
        train_sequences=data['train_sequences'],
        train_labels=data['train_labels'],
        num_classes=3,
        epochs=2
    )
    
    # Evaluate model
    loss, accuracy = evaluate_model(
        model=model,
        test_sequences=data['train_sequences'],  # Use training data for testing
        test_labels=data['train_labels'],
        class_names=['tech', 'sport', 'business']
    )
    
    # Check metrics
    assert isinstance(loss, float)
    assert isinstance(accuracy, float)
    assert 0 <= accuracy <= 1

def test_process_data_errors():
    """Test error handling in data processing."""
    # Test with non-existent file
    with pytest.raises(FileNotFoundError):
        process_data("nonexistent.csv")

def test_train_model_validation(processed_data):
    """Test model training with validation split."""
    data, tokenizer = processed_data
    
    # Train model with validation
    model, history = train_model(
        train_sequences=data['train_sequences'],
        train_labels=data['train_labels'],
        num_classes=3,
        epochs=2,
        validation_split=0.5
    )
    
    # Check validation metrics
    assert 'val_accuracy' in history
    assert 'val_loss' in history
    assert len(history['val_accuracy']) == 2
    assert len(history['val_loss']) == 2 