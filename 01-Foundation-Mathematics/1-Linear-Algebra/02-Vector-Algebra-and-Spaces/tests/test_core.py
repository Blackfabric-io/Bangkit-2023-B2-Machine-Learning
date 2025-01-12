import numpy as np
import pytest
from src.core.base import (
    layer_sizes,
    initialize_parameters,
    forward_propagation,
    compute_cost
)

def test_layer_sizes():
    X = np.array([[1, 2, 3], [4, 5, 6]])
    Y = np.array([[7, 8, 9]])
    
    n_x, n_y = layer_sizes(X, Y)
    
    assert n_x == 2
    assert n_y == 1
    
    with pytest.raises(TypeError):
        layer_sizes([1, 2, 3], Y)

def test_initialize_parameters():
    n_x, n_y = 2, 1
    parameters = initialize_parameters(n_x, n_y)
    
    assert isinstance(parameters, dict)
    assert "W" in parameters
    assert "b" in parameters
    assert parameters["W"].shape == (n_y, n_x)
    assert parameters["b"].shape == (n_y, 1)
    
    with pytest.raises(ValueError):
        initialize_parameters(-1, 1)

def test_forward_propagation():
    X = np.array([[1, 2], [3, 4]], dtype=float)
    parameters = {
        "W": np.array([[0.1, 0.2]], dtype=float),
        "b": np.array([[0.3]], dtype=float)
    }
    n_y = 1
    
    Y_hat = forward_propagation(X, parameters, n_y)
    
    assert Y_hat.shape == (n_y, X.shape[1])
    
    with pytest.raises(TypeError):
        forward_propagation([1, 2], parameters, n_y)
    
    with pytest.raises(TypeError):
        forward_propagation(X, {"W": [1, 2]}, n_y)

def test_compute_cost():
    Y_hat = np.array([[1, 2, 3]], dtype=float)
    Y = np.array([[1.1, 2.1, 2.9]], dtype=float)
    
    cost = compute_cost(Y_hat, Y)
    
    assert isinstance(cost, float)
    assert cost >= 0
    
    with pytest.raises(TypeError):
        compute_cost([1, 2], Y)
    
    with pytest.raises(ValueError):
        compute_cost(Y_hat, np.array([[1, 2]], dtype=float)) 