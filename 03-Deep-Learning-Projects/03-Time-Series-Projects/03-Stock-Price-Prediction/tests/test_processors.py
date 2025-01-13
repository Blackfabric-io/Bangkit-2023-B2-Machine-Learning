"""Test cases for data processing and model training."""

import numpy as np
import pandas as pd
import pytest
from src.processors.main import train_model, evaluate_model, compare_models
from src.utils.helpers import prepare_data, create_sequences
from src.core.base import RNNModel

def create_sample_data():
    """Create sample stock price data for testing."""
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    data = {
        'Close': np.random.normal(100, 10, 100),
        'Volume': np.random.normal(1000, 100, 100)
    }
    return pd.DataFrame(data, index=dates)

def test_data_preparation():
    """Test data preparation pipeline."""
    df = create_sample_data()
    sequence_length = 5
    
    X_train, y_train, X_val, y_val, scaler = prepare_data(
        df,
        sequence_length=sequence_length,
        train_split=0.8
    )
    
    # Check shapes
    assert len(X_train.shape) == 3  # (samples, sequence_length, features)
    assert len(y_train.shape) == 2  # (samples, features)
    assert X_train.shape[1] == sequence_length
    assert X_train.shape[2] == 1  # Single feature (Close price)
    
    # Check scaling
    assert np.all(X_train >= 0) and np.all(X_train <= 1)
    assert np.all(y_train >= 0) and np.all(y_train <= 1)

def test_sequence_creation():
    """Test sequence creation functionality."""
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
    sequence_length = 3
    
    X, y = create_sequences(data, sequence_length)
    
    assert X.shape == (7, 3, 1)  # (samples, sequence_length, features)
    assert y.shape == (7, 1)     # (samples, features)
    
    # Check first sequence
    np.testing.assert_array_equal(X[0], [[1], [2], [3]])
    np.testing.assert_array_equal(y[0], [4])

@pytest.mark.skip(reason="Requires actual data file")
def test_model_training():
    """Test model training workflow."""
    df = create_sample_data()
    df.to_csv('test_data.csv')
    
    results = train_model(
        RNNModel,
        'test_data.csv',
        sequence_length=5,
        epochs=2
    )
    
    assert 'model' in results
    assert 'history' in results
    assert 'scaler' in results
    assert 'validation_data' in results
    
    history = results['history']
    assert 'loss' in history
    assert 'val_loss' in history

@pytest.mark.skip(reason="Requires trained model")
def test_model_evaluation():
    """Test model evaluation functionality."""
    # Create synthetic data and model
    X = np.random.randn(10, 60, 1).astype(np.float32)
    y_true = np.random.randn(10, 1).astype(np.float32)
    
    model = RNNModel()
    model.compile_model()
    
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaler.fit(y_true)
    
    results = evaluate_model(model, X, y_true, scaler)
    
    assert 'predictions' in results
    assert 'mse' in results
    assert 'mae' in results
    assert results['predictions'].shape == y_true.shape

@pytest.mark.skip(reason="Requires actual data file")
def test_model_comparison():
    """Test model comparison functionality."""
    df = create_sample_data()
    df.to_csv('test_data.csv')
    
    results = compare_models(
        'test_data.csv',
        sequence_length=5,
        train_split=0.8
    )
    
    assert 'RNN' in results
    assert 'LSTM' in results
    assert 'BiLSTM' in results
    
    for model_results in results.values():
        assert 'training' in model_results
        assert 'evaluation' in model_results 