"""
Core implementation of neural network for digit recognition.
"""

import numpy as np
import numpy.typing as npt
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Type aliases
ArrayType = npt.NDArray[np.float64]

@dataclass
class NeuralNetConfig:
    """Configuration for neural network."""
    input_size: int
    hidden_sizes: List[int]
    output_size: int
    learning_rate: float = 0.1
    activation: str = 'relu'
    output_activation: str = 'softmax'
    weight_init: str = 'he'

class NeuralNetwork:
    """Neural network implementation for digit recognition."""
    
    def __init__(self, config: NeuralNetConfig):
        """Initialize neural network.
        
        Args:
            config: Network configuration parameters.
            
        Raises:
            ValueError: If parameters are invalid.
        """
        self._validate_config(config)
        self.config = config
        
        # Initialize weights and biases
        self.parameters = self._initialize_parameters()
        self.gradients: Dict[str, ArrayType] = {}
        self.cache: Dict[str, ArrayType] = {}
        
        logger.info("Initialized neural network with config: %s", config)
    
    def _validate_config(self, config: NeuralNetConfig) -> None:
        """Validate configuration parameters.
        
        Args:
            config: Configuration to validate.
            
        Raises:
            ValueError: If parameters are invalid.
        """
        if config.input_size <= 0:
            raise ValueError(f"Invalid input size: {config.input_size}")
        if any(size <= 0 for size in config.hidden_sizes):
            raise ValueError(f"Invalid hidden sizes: {config.hidden_sizes}")
        if config.output_size <= 0:
            raise ValueError(f"Invalid output size: {config.output_size}")
        if not 0 < config.learning_rate <= 1:
            raise ValueError(f"Invalid learning rate: {config.learning_rate}")
        if config.activation not in ['relu', 'tanh']:
            raise ValueError(f"Invalid activation: {config.activation}")
        if config.output_activation not in ['softmax']:
            raise ValueError(f"Invalid output activation: {config.output_activation}")
        if config.weight_init not in ['he', 'xavier']:
            raise ValueError(f"Invalid weight initialization: {config.weight_init}")
    
    def _initialize_parameters(self) -> Dict[str, ArrayType]:
        """Initialize network parameters using specified initialization method.
        
        Returns:
            Dictionary containing weights and biases.
        """
        parameters = {}
        layer_dims = [self.config.input_size] + \
                    self.config.hidden_sizes + \
                    [self.config.output_size]
        
        for l in range(1, len(layer_dims)):
            if self.config.weight_init == 'he':
                scale = np.sqrt(2.0 / layer_dims[l-1])
            else:  # xavier
                scale = np.sqrt(1.0 / layer_dims[l-1])
            
            parameters[f'W{l}'] = np.random.randn(
                layer_dims[l], layer_dims[l-1]) * scale
            parameters[f'b{l}'] = np.zeros((layer_dims[l], 1))
            
        return parameters
    
    def _relu(self, Z: ArrayType) -> ArrayType:
        """ReLU activation function."""
        return np.maximum(0, Z)
    
    def _relu_backward(self, dA: ArrayType, Z: ArrayType) -> ArrayType:
        """Backward pass for ReLU activation."""
        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0
        return dZ
    
    def _tanh(self, Z: ArrayType) -> ArrayType:
        """Tanh activation function."""
        return np.tanh(Z)
    
    def _tanh_backward(self, dA: ArrayType, Z: ArrayType) -> ArrayType:
        """Backward pass for tanh activation."""
        return dA * (1 - np.power(self._tanh(Z), 2))
    
    def _softmax(self, Z: ArrayType) -> ArrayType:
        """Softmax activation function."""
        expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
        return expZ / np.sum(expZ, axis=0, keepdims=True)
    
    def _forward_step(self, A_prev: ArrayType, W: ArrayType,
                     b: ArrayType, activation: str) -> Tuple[ArrayType, ArrayType]:
        """Single forward propagation step.
        
        Args:
            A_prev: Activations from previous layer.
            W: Weight matrix.
            b: Bias vector.
            activation: Activation function to use.
            
        Returns:
            Tuple of (activation output, cache for backward pass).
        """
        Z = np.dot(W, A_prev) + b
        
        if activation == 'relu':
            A = self._relu(Z)
        elif activation == 'tanh':
            A = self._tanh(Z)
        else:  # softmax
            A = self._softmax(Z)
            
        return A, Z
    
    def forward_propagation(self, X: ArrayType) -> ArrayType:
        """Forward propagation through the network.
        
        Args:
            X: Input data of shape (input_size, m).
            
        Returns:
            Output probabilities.
            
        Raises:
            ValueError: If input shape is invalid.
        """
        if X.shape[0] != self.config.input_size:
            raise ValueError(
                f"Input shape {X.shape} does not match "
                f"expected shape ({self.config.input_size}, m)"
            )
        
        self.cache = {'A0': X}
        A = X
        
        # Hidden layers
        for l in range(1, len(self.config.hidden_sizes) + 1):
            A, Z = self._forward_step(
                A, self.parameters[f'W{l}'],
                self.parameters[f'b{l}'],
                self.config.activation
            )
            self.cache[f'A{l}'] = A
            self.cache[f'Z{l}'] = Z
        
        # Output layer
        A, Z = self._forward_step(
            A, self.parameters[f'W{len(self.parameters)//2}'],
            self.parameters[f'b{len(self.parameters)//2}'],
            self.config.output_activation
        )
        self.cache[f'A{len(self.parameters)//2}'] = A
        self.cache[f'Z{len(self.parameters)//2}'] = Z
        
        return A
    
    def _backward_step(self, dA: ArrayType, W: ArrayType,
                      Z: ArrayType, A_prev: ArrayType,
                      activation: str) -> Tuple[ArrayType, ArrayType, ArrayType]:
        """Single backward propagation step.
        
        Args:
            dA: Gradient of cost with respect to current layer's activation.
            W: Weight matrix.
            Z: Pre-activation cache.
            A_prev: Previous layer's activation.
            activation: Activation function used.
            
        Returns:
            Tuple of (gradient w.r.t previous activation,
                     gradient w.r.t weights,
                     gradient w.r.t bias).
        """
        m = A_prev.shape[1]
        
        if activation == 'relu':
            dZ = self._relu_backward(dA, Z)
        elif activation == 'tanh':
            dZ = self._tanh_backward(dA, Z)
        else:  # softmax
            dZ = dA  # Assuming dA is already the gradient from softmax
        
        dW = np.dot(dZ, A_prev.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m
        dA_prev = np.dot(W.T, dZ)
        
        return dA_prev, dW, db
    
    def backward_propagation(self, Y: ArrayType) -> None:
        """Backward propagation through the network.
        
        Args:
            Y: True labels in one-hot format.
            
        Raises:
            ValueError: If label shape is invalid.
        """
        if Y.shape[0] != self.config.output_size:
            raise ValueError(
                f"Label shape {Y.shape} does not match "
                f"expected shape ({self.config.output_size}, m)"
            )
        
        m = Y.shape[1]
        L = len(self.parameters) // 2
        
        # Initialize gradients
        self.gradients = {}
        
        # Output layer
        dAL = self.cache[f'A{L}'] - Y
        
        # Backward pass through layers
        dA = dAL
        for l in reversed(range(1, L + 1)):
            dA, dW, db = self._backward_step(
                dA,
                self.parameters[f'W{l}'],
                self.cache[f'Z{l}'],
                self.cache[f'A{l-1}'],
                self.config.activation if l < L else self.config.output_activation
            )
            self.gradients[f'dW{l}'] = dW
            self.gradients[f'db{l}'] = db
    
    def update_parameters(self) -> None:
        """Update network parameters using computed gradients."""
        L = len(self.parameters) // 2
        
        for l in range(1, L + 1):
            self.parameters[f'W{l}'] -= \
                self.config.learning_rate * self.gradients[f'dW{l}']
            self.parameters[f'b{l}'] -= \
                self.config.learning_rate * self.gradients[f'db{l}']
    
    def compute_cost(self, Y: ArrayType) -> float:
        """Compute cross-entropy cost.
        
        Args:
            Y: True labels in one-hot format.
            
        Returns:
            Cost value.
        """
        m = Y.shape[1]
        AL = self.cache[f'A{len(self.parameters)//2}']
        
        # Compute cross-entropy loss
        cost = -np.sum(Y * np.log(AL + 1e-8)) / m
        return float(cost)
    
    def predict(self, X: ArrayType) -> ArrayType:
        """Make predictions for input data.
        
        Args:
            X: Input data of shape (input_size, m).
            
        Returns:
            Predicted digit labels.
        """
        probabilities = self.forward_propagation(X)
        return np.argmax(probabilities, axis=0)
    
    def save_parameters(self, filepath: str) -> None:
        """Save network parameters to file.
        
        Args:
            filepath: Path to save file.
        """
        np.save(filepath, self.parameters)
        logger.info("Saved parameters to %s", filepath)
    
    def load_parameters(self, filepath: str) -> None:
        """Load network parameters from file.
        
        Args:
            filepath: Path to load file.
            
        Raises:
            ValueError: If loaded parameters have wrong shapes.
        """
        loaded = np.load(filepath, allow_pickle=True).item()
        
        # Validate shapes
        for key, value in loaded.items():
            if value.shape != self.parameters[key].shape:
                raise ValueError(
                    f"Loaded parameter {key} shape {value.shape} does not match "
                    f"expected shape {self.parameters[key].shape}"
                )
        
        self.parameters = loaded
        logger.info("Loaded parameters from %s", filepath) 