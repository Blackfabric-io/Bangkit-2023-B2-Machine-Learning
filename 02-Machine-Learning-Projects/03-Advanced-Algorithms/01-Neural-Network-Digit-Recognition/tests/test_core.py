"""
Unit tests for neural network implementation.
"""

import pytest
import numpy as np
from src.core.base import NeuralNetwork, NeuralNetConfig

def test_neural_network_initialization():
    """Test initialization of neural network."""
    config = NeuralNetConfig(
        input_size=784,
        hidden_sizes=[128, 64],
        output_size=10
    )
    model = NeuralNetwork(config)
    
    # Check parameters initialization
    assert len(model.parameters) == 6  # 3 layers * (W, b)
    assert model.parameters['W1'].shape == (128, 784)
    assert model.parameters['b1'].shape == (128, 1)
    assert model.parameters['W2'].shape == (64, 128)
    assert model.parameters['b2'].shape == (64, 1)
    assert model.parameters['W3'].shape == (10, 64)
    assert model.parameters['b3'].shape == (10, 1)

def test_neural_network_invalid_config():
    """Test neural network with invalid configuration."""
    # Test invalid input size
    with pytest.raises(ValueError):
        config = NeuralNetConfig(input_size=0, hidden_sizes=[128], output_size=10)
        NeuralNetwork(config)
    
    # Test invalid hidden sizes
    with pytest.raises(ValueError):
        config = NeuralNetConfig(input_size=784, hidden_sizes=[-1], output_size=10)
        NeuralNetwork(config)
    
    # Test invalid output size
    with pytest.raises(ValueError):
        config = NeuralNetConfig(input_size=784, hidden_sizes=[128], output_size=0)
        NeuralNetwork(config)
    
    # Test invalid learning rate
    with pytest.raises(ValueError):
        config = NeuralNetConfig(
            input_size=784, hidden_sizes=[128], output_size=10,
            learning_rate=1.5
        )
        NeuralNetwork(config)

def test_activation_functions():
    """Test activation functions."""
    config = NeuralNetConfig(
        input_size=784,
        hidden_sizes=[128],
        output_size=10
    )
    model = NeuralNetwork(config)
    
    # Test ReLU
    Z = np.array([[-1, 0, 1], [2, -3, 4]])
    A = model._relu(Z)
    assert np.array_equal(A, np.array([[0, 0, 1], [2, 0, 4]]))
    
    # Test softmax
    Z = np.array([[1, 2], [3, 4], [5, 6]])
    A = model._softmax(Z)
    assert A.shape == Z.shape
    assert np.allclose(np.sum(A, axis=0), [1, 1])
    assert np.all(A >= 0) and np.all(A <= 1)

def test_forward_propagation():
    """Test forward propagation."""
    config = NeuralNetConfig(
        input_size=3,
        hidden_sizes=[4],
        output_size=2
    )
    model = NeuralNetwork(config)
    
    # Test with valid input
    X = np.random.randn(3, 5)
    A = model.forward_propagation(X)
    assert A.shape == (2, 5)
    assert np.allclose(np.sum(A, axis=0), np.ones(5))
    
    # Test with invalid input shape
    with pytest.raises(ValueError):
        X_invalid = np.random.randn(4, 5)
        model.forward_propagation(X_invalid)

def test_backward_propagation():
    """Test backward propagation."""
    config = NeuralNetConfig(
        input_size=3,
        hidden_sizes=[4],
        output_size=2
    )
    model = NeuralNetwork(config)
    
    # Forward pass
    X = np.random.randn(3, 5)
    A = model.forward_propagation(X)
    
    # Test with valid labels
    Y = np.zeros((2, 5))
    Y[0, [0, 2, 4]] = 1
    Y[1, [1, 3]] = 1
    model.backward_propagation(Y)
    
    # Check gradients initialization
    assert len(model.gradients) == len(model.parameters)
    for key in model.parameters:
        assert model.gradients[f'd{key}'].shape == model.parameters[key].shape
    
    # Test with invalid label shape
    with pytest.raises(ValueError):
        Y_invalid = np.zeros((3, 5))
        model.backward_propagation(Y_invalid)

def test_parameter_update():
    """Test parameter update."""
    config = NeuralNetConfig(
        input_size=3,
        hidden_sizes=[4],
        output_size=2,
        learning_rate=0.1
    )
    model = NeuralNetwork(config)
    
    # Store initial parameters
    initial_params = {
        key: value.copy() for key, value in model.parameters.items()
    }
    
    # Forward and backward pass
    X = np.random.randn(3, 5)
    Y = np.zeros((2, 5))
    Y[0, [0, 2, 4]] = 1
    Y[1, [1, 3]] = 1
    
    model.forward_propagation(X)
    model.backward_propagation(Y)
    model.update_parameters()
    
    # Check that parameters have been updated
    for key in model.parameters:
        assert not np.array_equal(model.parameters[key], initial_params[key])

def test_cost_computation():
    """Test cost computation."""
    config = NeuralNetConfig(
        input_size=3,
        hidden_sizes=[4],
        output_size=2
    )
    model = NeuralNetwork(config)
    
    # Forward pass
    X = np.random.randn(3, 5)
    model.forward_propagation(X)
    
    # Test with valid labels
    Y = np.zeros((2, 5))
    Y[0, [0, 2, 4]] = 1
    Y[1, [1, 3]] = 1
    cost = model.compute_cost(Y)
    
    assert isinstance(cost, float)
    assert cost >= 0

def test_prediction():
    """Test prediction."""
    config = NeuralNetConfig(
        input_size=3,
        hidden_sizes=[4],
        output_size=2
    )
    model = NeuralNetwork(config)
    
    # Test predictions
    X = np.random.randn(3, 5)
    predictions = model.predict(X)
    
    assert predictions.shape == (5,)
    assert np.all(predictions >= 0) and np.all(predictions < 2)

def test_parameter_saving_loading(tmp_path):
    """Test parameter saving and loading."""
    config = NeuralNetConfig(
        input_size=3,
        hidden_sizes=[4],
        output_size=2
    )
    model = NeuralNetwork(config)
    
    # Save parameters
    save_path = tmp_path / "params.npy"
    model.save_parameters(save_path)
    
    # Create new model and load parameters
    new_model = NeuralNetwork(config)
    new_model.load_parameters(save_path)
    
    # Check that parameters match
    for key in model.parameters:
        assert np.array_equal(model.parameters[key], new_model.parameters[key])
    
    # Test loading with wrong shapes
    wrong_config = NeuralNetConfig(
        input_size=4,
        hidden_sizes=[5],
        output_size=3
    )
    wrong_model = NeuralNetwork(wrong_config)
    
    with pytest.raises(ValueError):
        wrong_model.load_parameters(save_path) 