"""Test cases for data processing and model training."""

import numpy as np
import pandas as pd
import pytest
from src.processors.main import train_model, evaluate_model, simulate_real_time
from src.utils.helpers import prepare_data, create_windows
from src.core.base import RealTimeModel

def create_sample_data():
    """Create sample time series data for testing."""
    dates = pd.date_range('2023-01-01', periods=1000, freq='H')
    data = {
        'temperature': np.sin(np.linspace(0, 10*np.pi, 1000)) + np.random.normal(0, 0.1, 1000),
        'humidity': np.cos(np.linspace(0, 10*np.pi, 1000)) + np.random.normal(0, 0.1, 1000)
    }
    df = pd.DataFrame(data, index=dates)
    df.index.name = 'timestamp'
    return df

def test_data_preparation():
    """Test data preparation pipeline."""
    df = create_sample_data()
    input_width = 24
    label_width = 1
    shift = 1
    
    train_ds, val_ds, scaler = prepare_data(
        df,
        input_width=input_width,
        label_width=label_width,
        shift=shift,
        train_split=0.8
    )
    
    # Check dataset structure
    for x, y in train_ds.take(1):
        assert x.shape[1] == input_width
        assert y.shape[1] == label_width
        assert x.shape[2] == df.shape[1]  # Number of features
        assert y.shape[2] == df.shape[1]

def test_window_creation():
    """Test window creation functionality."""
    data = np.random.randn(100, 2)  # 100 timesteps, 2 features
    window_size = 24
    stride = 1
    
    windows = create_windows(data, window_size, stride)
    
    expected_samples = (len(data) - window_size) // stride + 1
    assert windows.shape == (expected_samples, window_size, 2)

@pytest.mark.skip(reason="Requires actual data file")
def test_model_training():
    """Test model training workflow."""
    df = create_sample_data()
    df.to_csv('test_data.csv')
    
    results = train_model(
        'test_data.csv',
        input_width=24,
        label_width=1,
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
    df = create_sample_data()
    train_ds, val_ds, scaler = prepare_data(df)
    
    model = RealTimeModel()
    model.build_model(num_features=2)
    model.compile_model()
    
    results = evaluate_model(
        model,
        val_ds,
        scaler,
        feature_names=['temperature', 'humidity'],
        return_uncertainty=True
    )
    
    assert 'predictions' in results
    assert 'true_values' in results
    assert 'std' in results
    assert 'metrics' in results
    
    metrics = results['metrics']
    assert 'MSE' in metrics.columns
    assert 'MAE' in metrics.columns
    assert 'RMSE' in metrics.columns

def test_real_time_simulation():
    """Test real-time simulation functionality."""
    df = create_sample_data()
    model = RealTimeModel(input_width=24)
    model.build_model(num_features=2)
    model.compile_model()
    
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(df.values)
    
    results = simulate_real_time(
        model,
        df,
        scaler,
        window_size=24,
        n_steps=10
    )
    
    assert len(results) == 10
    assert 'step' in results.columns
    assert 'true' in results.columns
    assert 'predicted' in results.columns
    assert 'std' in results.columns
    assert 'loss' in results.columns 