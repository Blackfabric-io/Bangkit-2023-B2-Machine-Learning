"""Tests for the core HousePriceModel."""

import pytest
import numpy as np
from src.core import HousePriceModel

def test_model_initialization():
    """Test that model can be initialized."""
    model = HousePriceModel()
    assert model is not None
    assert model.model is not None

def test_data_preparation():
    """Test data preparation returns correct shapes."""
    bedrooms, prices = HousePriceModel.prepare_data()
    assert isinstance(bedrooms, np.ndarray)
    assert isinstance(prices, np.ndarray)
    assert bedrooms.shape == prices.shape
    assert len(bedrooms.shape) == 1

def test_model_training():
    """Test that model can be trained."""
    model = HousePriceModel()
    bedrooms, prices = model.prepare_data()
    history = model.train(bedrooms, prices, epochs=10, verbose=0)
    assert history is not None
    assert isinstance(history, tf.keras.callbacks.History)
    assert 'loss' in history.history

def test_model_prediction():
    """Test that model can make predictions."""
    model = HousePriceModel()
    bedrooms, prices = model.prepare_data()
    model.train(bedrooms, prices, epochs=10, verbose=0)
    
    test_input = np.array([7.0])
    prediction = model.predict(test_input)
    
    assert prediction is not None
    assert isinstance(prediction, np.ndarray)
    assert prediction.shape == (1, 1)
    # Basic sanity check - prediction should be positive
    assert prediction[0, 0] > 0 