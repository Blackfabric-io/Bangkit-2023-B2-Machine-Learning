import numpy as np
import pytest
from src.processors.main import (
    nn_model,
    normalize_data,
    predict
)

def test_nn_model():
    X = np.array([[1, 2], [3, 4]], dtype=float)
    Y = np.array([[5, 6]], dtype=float)
    
    parameters = nn_model(X, Y, num_iterations=2, print_cost=False)
    
    assert isinstance(parameters, dict)
    assert "W" in parameters
    assert "b" in parameters
    assert parameters["W"].shape == (1, 2)
    assert parameters["b"].shape == (1, 1)
    
    with pytest.raises(TypeError):
        nn_model([1, 2], Y)
    
    with pytest.raises(ValueError):
        nn_model(X, Y, num_iterations=-1)

def test_normalize_data():
    X = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
    
    X_norm, X_mean, X_std = normalize_data(X)
    
    assert X_norm.shape == X.shape
    assert X_mean.shape == (2, 1)
    assert X_std.shape == (2, 1)
    
    """
    Check normalization results in zero mean and unit variance
    Normalization should make the data have a mean of 0 (centered) and a variance of 1 (consistent spread).
    Think of it like adjusting a team's performance so everyone is on the same level!
    """
    assert np.allclose(np.mean(X_norm, axis=1), np.zeros(2))
    assert np.allclose(np.std(X_norm, axis=1), np.ones(2))
    
    with pytest.raises(TypeError):
        normalize_data([1, 2, 3])

def test_predict():
    X = np.array([[1, 2]], dtype=float)
    parameters = {
        "W": np.array([[0.1, 0.2]], dtype=float),
        "b": np.array([[0.3]], dtype=float)
    }
    
    # Test without normalization
    Y_pred = predict(X, parameters)
    assert Y_pred.shape == (1, 2)
    
    # Test with normalization
    X_mean = np.array([[1.5]], dtype=float)
    X_std = np.array([[0.5]], dtype=float)
    Y_mean = 2.0
    Y_std = 1.0
    
    Y_pred = predict(X, parameters, X_mean, X_std, Y_mean, Y_std)
    assert Y_pred.shape == (1, 2)
    
    with pytest.raises(TypeError):
        predict([1, 2], parameters)
    
    with pytest.raises(TypeError):
        predict(X, {"W": [1, 2]}) 