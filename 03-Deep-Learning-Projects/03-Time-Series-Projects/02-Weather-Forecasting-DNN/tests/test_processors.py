"""Test cases for data processing and model training."""

import numpy as np
import pandas as pd
import pytest
from src.processors.main import prepare_data, train_model, evaluate_model
from src.utils.helpers import normalize_data, denormalize_data

def create_sample_data():
    """Create sample weather data for testing."""
    dates = pd.date_range('2023-01-01', periods=100, freq='H')
    data = {
        'Temperature': np.random.normal(20, 5, 100),
        'Humidity': np.random.normal(60, 10, 100)
    }
    return pd.DataFrame(data, index=dates)

def test_data_normalization():
    """Test data normalization and denormalization."""
    df = create_sample_data()
    
    # Test normalization
    normalized_df, feature_ranges = normalize_data(df)
    
    assert normalized_df.shape == df.shape
    assert all(normalized_df.min() >= 0)
    assert all(normalized_df.max() <= 1)
    
    # Test denormalization
    denormalized = denormalize_data(
        np.array(normalized_df),
        feature_ranges,
        df.columns.tolist()
    )
    
    np.testing.assert_array_almost_equal(df.values, denormalized.values)

@pytest.mark.skip(reason="Requires actual data file")
def test_data_preparation():
    """Test data preparation pipeline."""
    data_path = "weather_data.csv"
    train_ds, val_ds, feature_ranges = prepare_data(data_path)
    
    # Check that datasets are created
    assert train_ds is not None
    assert val_ds is not None
    assert len(feature_ranges) > 0
    
    # Check dataset structure
    for inputs, labels in train_ds.take(1):
        assert len(inputs.shape) == 3  # (batch, time, features)
        assert len(labels.shape) == 3

@pytest.mark.skip(reason="Requires actual training")
def test_model_training():
    """Test model training workflow."""
    # Create synthetic data
    df = create_sample_data()
    normalized_df, feature_ranges = normalize_data(df)
    data = np.array(normalized_df)
    
    # Split data
    train_size = int(0.8 * len(data))
    train_data = data[:train_size]
    val_data = data[train_size:]
    
    # Create model and train
    from src.core.base import WeatherModel
    model = WeatherModel()
    train_ds = model.make_dataset(train_data)
    val_ds = model.make_dataset(val_data, shuffle=False)
    
    trained_model, history = train_model(
        train_ds,
        val_ds,
        num_features=2,
        max_epochs=2
    )
    
    assert 'loss' in history
    assert 'val_loss' in history
    assert len(history['loss']) <= 2

@pytest.mark.skip(reason="Requires trained model")
def test_model_evaluation():
    """Test model evaluation functionality."""
    # Create synthetic data and model
    df = create_sample_data()
    normalized_df, feature_ranges = normalize_data(df)
    data = np.array(normalized_df)
    
    model = WeatherModel()
    model.build_model(num_features=2)
    
    # Create validation dataset
    val_ds = model.make_dataset(data, shuffle=False)
    
    # Evaluate model
    true_df, pred_df = evaluate_model(
        model,
        val_ds,
        feature_ranges,
        df.columns.tolist()
    )
    
    assert true_df.shape[1] == df.shape[1]
    assert pred_df.shape[1] == df.shape[1] 