"""Core implementation of neural network for digit recognition.

This module provides the core neural network functionality including:
- Neural network model with configurable architecture
- ReLU activation function
- Softmax output layer
- Forward/backward propagation
- Cost computation
"""

import numpy as np
import tensorflow as tf
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Type aliases
ArrayType = np.ndarray 

@dataclass
class NeuralNetConfig:
    """Configuration for neural network architecture and training."""
    input_size: int = 400
    hidden_sizes: List[int] = None
    output_size: int = 10
    learning_rate: float = 0.001
    num_iterations: int = 1000
    batch_size: int = 32
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.hidden_sizes is None:
            self.hidden_sizes = [25, 15]
        if self.input_size <= 0:
            raise ValueError("input_size must be positive")
        if any(size <= 0 for size in self.hidden_sizes):
            raise ValueError("hidden_sizes must be positive")
        if self.output_size <= 0:
            raise ValueError("output_size must be positive")

class NeuralNetwork:
    """Neural network implementation for digit recognition."""
    
    def __init__(self, config: NeuralNetConfig):
        """Initialize neural network with given configuration.
        
        Args:
            config: Neural network configuration
        """
        self.config = config
        self.parameters = {}
        self.initialize_parameters()
        logger.info(f"Initialized neural network with architecture: {[config.input_size] + config.hidden_sizes + [config.output_size]}")
        
    def initialize_parameters(self) -> None:
        """Initialize network parameters using He initialization."""
        layer_sizes = [self.config.input_size] + self.config.hidden_sizes + [self.config.output_size]
        
        for l in range(1, len(layer_sizes)):
            self.parameters[f'W{l}'] = np.random.randn(layer_sizes[l], layer_sizes[l-1]) * np.sqrt(2/layer_sizes[l-1])
            self.parameters[f'b{l}'] = np.zeros((layer_sizes[l], 1))
            
        logger.debug("Initialized network parameters")
    
    @staticmethod
    def relu(Z: ArrayType) -> ArrayType:
        """ReLU activation function."""
        return np.maximum(0, Z)
    
    @staticmethod
    def relu_derivative(Z: ArrayType) -> ArrayType:
        """Derivative of ReLU activation function."""
        return np.where(Z > 0, 1, 0)
    
    @staticmethod
    def softmax(Z: ArrayType) -> ArrayType:
        """Softmax activation function."""
        exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
        return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
    
    def forward_propagation(self, X: ArrayType) -> Tuple[Dict[str, ArrayType], Dict[str, ArrayType]]:
        """Forward propagation step.
        
        Args:
            X: Input data of shape (input_size, m)
            
        Returns:
            Tuple of activations and cache for backpropagation
        """
        activations = {'A0': X}
        cache = {}
        
        # Hidden layers with ReLU
        for l in range(1, len(self.config.hidden_sizes) + 1):
            Z = np.dot(self.parameters[f'W{l}'], activations[f'A{l-1}']) + self.parameters[f'b{l}']
            cache[f'Z{l}'] = Z
            activations[f'A{l}'] = self.relu(Z)
            
        # Output layer with softmax
        l = len(self.config.hidden_sizes) + 1
        Z = np.dot(self.parameters[f'W{l}'], activations[f'A{l-1}']) + self.parameters[f'b{l}']
        cache[f'Z{l}'] = Z
        activations[f'A{l}'] = self.softmax(Z)
        
        return activations, cache
    
    def compute_cost(self, AL: ArrayType, Y: ArrayType) -> float:
        """Compute cross-entropy cost.
        
        Args:
            AL: Output of the neural network
            Y: True labels
            
        Returns:
            Cost value
        """
        m = Y.shape[1]
        cost = -1/m * np.sum(Y * np.log(AL + 1e-8))
        return float(cost)
    
    def backward_propagation(self, Y: ArrayType, activations: Dict[str, ArrayType], 
                           cache: Dict[str, ArrayType]) -> Dict[str, ArrayType]:
        """Backward propagation to compute gradients.
        
        Args:
            Y: True labels
            activations: Activations from forward propagation
            cache: Cache from forward propagation
            
        Returns:
            Dictionary containing gradients
        """
        m = Y.shape[1]
        L = len(self.config.hidden_sizes) + 1
        gradients = {}
        
        # Output layer
        dZ = activations[f'A{L}'] - Y
        gradients[f'dW{L}'] = 1/m * np.dot(dZ, activations[f'A{L-1}'].T)
        gradients[f'db{L}'] = 1/m * np.sum(dZ, axis=1, keepdims=True)
        
        # Hidden layers
        for l in reversed(range(1, L)):
            dA = np.dot(self.parameters[f'W{l+1}'].T, dZ)
            dZ = dA * self.relu_derivative(cache[f'Z{l}'])
            gradients[f'dW{l}'] = 1/m * np.dot(dZ, activations[f'A{l-1}'].T)
            gradients[f'db{l}'] = 1/m * np.sum(dZ, axis=1, keepdims=True)
            
        return gradients
    
    def update_parameters(self, gradients: Dict[str, ArrayType]) -> None:
        """Update network parameters using gradient descent.
        
        Args:
            gradients: Gradients computed in backward propagation
        """
        L = len(self.config.hidden_sizes) + 1
        for l in range(1, L + 1):
            self.parameters[f'W{l}'] -= self.config.learning_rate * gradients[f'dW{l}']
            self.parameters[f'b{l}'] -= self.config.learning_rate * gradients[f'db{l}']
    
    def predict(self, X: ArrayType) -> ArrayType:
        """Make predictions using the trained network.
        
        Args:
            X: Input data
            
        Returns:
            Predicted labels
        """
        activations, _ = self.forward_propagation(X)
        predictions = np.argmax(activations[f'A{len(self.config.hidden_sizes) + 1}'], axis=0)
        return predictions
    
    def save_model(self, filepath: str) -> None:
        """Save model parameters to file.
        
        Args:
            filepath: Path to save the model
        """
        try:
            np.save(filepath, self.parameters)
            logger.info(f"Model saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
    
    def load_model(self, filepath: str) -> None:
        """Load model parameters from file.
        
        Args:
            filepath: Path to load the model from
        """
        try:
            self.parameters = np.load(filepath, allow_pickle=True).item()
            logger.info(f"Model loaded from {filepath}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise 