"""
Unit tests for digit recognition processor.
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
from src.processors.main import DigitRecognizer

def create_synthetic_data(n_samples: int = 100,
                        input_size: int = 784,
                        n_classes: int = 10) -> tuple:
    """Create synthetic data for testing.
    
    Args:
        n_samples: Number of samples.
        input_size: Size of input features.
        n_classes: Number of classes.
        
    Returns:
        Tuple of (X_train, y_train, X_test, y_test).
    """
    # Create training data
    X_train = np.random.randn(n_samples, input_size)
    y_train = np.random.randint(0, n_classes, size=n_samples)
    
    # Create test data
    X_test = np.random.randn(n_samples // 2, input_size)
    y_test = np.random.randint(0, n_classes, size=n_samples // 2)
    
    return X_train, y_train, X_test, y_test

def test_digit_recognizer_initialization():
    """Test initialization of digit recognizer."""
    model = DigitRecognizer(
        input_size=784,
        hidden_sizes=[128, 64],
        output_size=10
    )
    
    assert model.parameters['input_size'] == 784
    assert model.parameters['hidden_sizes'] == [128, 64]
    assert model.parameters['output_size'] == 10
    assert isinstance(model.network, NeuralNetwork)

def test_digit_recognizer_data_loading(tmp_path):
    """Test data loading functionality."""
    # Create synthetic data
    X_train, y_train, X_test, y_test = create_synthetic_data()
    
    # Save synthetic data
    data_dir = tmp_path / 'data'
    data_dir.mkdir()
    np.save(data_dir / 'X_train.npy', X_train)
    np.save(data_dir / 'y_train.npy', y_train)
    np.save(data_dir / 'X_test.npy', X_test)
    np.save(data_dir / 'y_test.npy', y_test)
    
    # Initialize model and load data
    model = DigitRecognizer()
    model.load_data(data_dir)
    
    # Check data shapes
    assert model.X_train.shape[0] == 784
    assert model.X_train.shape[1] == 100
    assert model.y_train.shape[0] == 10
    assert model.y_train.shape[1] == 100
    assert model.X_test.shape[0] == 784
    assert model.X_test.shape[1] == 50
    assert model.y_test.shape[0] == 10
    assert model.y_test.shape[1] == 50

def test_digit_recognizer_training():
    """Test training process."""
    # Create synthetic data
    X_train, y_train, X_test, y_test = create_synthetic_data()
    
    # Initialize model
    model = DigitRecognizer(
        input_size=784,
        hidden_sizes=[32],  # Smaller network for testing
        output_size=10
    )
    
    # Set data manually
    model.X_train = X_train.T
    model.y_train = np.eye(10)[y_train].T
    model.X_test = X_test.T
    model.y_test = np.eye(10)[y_test].T
    
    # Train for a few iterations
    results = model.train(
        n_iterations=10,
        batch_size=32,
        eval_interval=5
    )
    
    assert 'costs' in results
    assert len(results['costs']) == 10
    assert 'final_accuracy' in results
    assert 0 <= results['final_accuracy'] <= 1
    assert 'metrics' in results

def test_digit_recognizer_evaluation():
    """Test evaluation functionality."""
    # Create synthetic data
    X_train, y_train, X_test, y_test = create_synthetic_data()
    
    # Initialize model
    model = DigitRecognizer(
        input_size=784,
        hidden_sizes=[32],  # Smaller network for testing
        output_size=10
    )
    
    # Set data manually
    model.X_train = X_train.T
    model.y_train = np.eye(10)[y_train].T
    model.X_test = X_test.T
    model.y_test = np.eye(10)[y_test].T
    
    # Evaluate model
    metrics = model.evaluate(plot_results=False)
    
    assert 'accuracy' in metrics
    assert 0 <= metrics['accuracy'] <= 1
    
    # Check per-class metrics
    for i in range(10):
        assert f'precision_{i}' in metrics
        assert f'recall_{i}' in metrics
        assert f'f1_{i}' in metrics

def test_digit_recognizer_prediction():
    """Test prediction functionality."""
    # Create synthetic data
    X_train, y_train, X_test, y_test = create_synthetic_data()
    
    # Initialize model
    model = DigitRecognizer(
        input_size=784,
        hidden_sizes=[32],  # Smaller network for testing
        output_size=10
    )
    
    # Make predictions
    predictions = model.predict(X_test.T)
    
    assert predictions.shape == (50,)  # Test set size
    assert np.all(predictions >= 0) and np.all(predictions < 10)

def test_digit_recognizer_save_load(tmp_path):
    """Test model saving and loading."""
    # Create synthetic data
    X_train, y_train, X_test, y_test = create_synthetic_data()
    
    # Initialize and train model
    model = DigitRecognizer(
        input_size=784,
        hidden_sizes=[32],  # Smaller network for testing
        output_size=10
    )
    
    # Set data manually
    model.X_train = X_train.T
    model.y_train = np.eye(10)[y_train].T
    model.X_test = X_test.T
    model.y_test = np.eye(10)[y_test].T
    
    # Train briefly
    model.train(n_iterations=5, batch_size=32)
    
    # Save model
    save_dir = tmp_path / 'model'
    model.save(save_dir)
    
    # Load model
    loaded_model = DigitRecognizer.load(save_dir)
    
    # Compare parameters
    assert model.parameters == loaded_model.parameters
    
    # Compare predictions
    X = np.random.randn(784, 10)
    assert np.array_equal(
        model.predict(X),
        loaded_model.predict(X)
    )

def test_digit_recognizer_error_handling():
    """Test error handling."""
    model = DigitRecognizer()
    
    # Test training without data
    with pytest.raises(ValueError):
        model.train()
    
    # Test evaluation without data
    with pytest.raises(ValueError):
        model.evaluate()
    
    # Test loading with invalid path
    with pytest.raises(FileNotFoundError):
        DigitRecognizer.load('nonexistent_path') 