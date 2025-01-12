"""
Core functionality for logistic regression model.
"""

import numpy as np
from typing import Tuple, Optional, List
import numpy.typing as npt

ArrayType = npt.NDArray[np.float64]

class LogisticRegression:
    """Logistic regression model implementation.
    
    This class implements binary logistic regression using gradient descent optimization.
    The model fits a sigmoid function to predict probabilities of class membership.
    """
    
    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 1000):
        """Initialize the model.
        
        Args:
            learning_rate: Learning rate for gradient descent.
            n_iterations: Number of gradient descent iterations.
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights: Optional[ArrayType] = None
        self.bias: Optional[float] = None
        self.cost_history: List[float] = []
        
    def sigmoid(self, z: ArrayType) -> ArrayType:
        """Compute sigmoid function.
        
        Args:
            z: Input values.
            
        Returns:
            Sigmoid of input values.
        """
        return 1 / (1 + np.exp(-z))
    
    def compute_cost(self, X: ArrayType, y: ArrayType) -> float:
        """Compute binary cross-entropy cost.
        
        Args:
            X: Training features of shape (m, n).
            y: Target values of shape (m,).
            
        Returns:
            Cost value.
            
        Raises:
            ValueError: If model is not fitted or input shapes don't match.
        """
        if self.weights is None or self.bias is None:
            raise ValueError("Model must be fitted before computing cost")
            
        m = X.shape[0]
        z = np.dot(X, self.weights) + self.bias
        h = self.sigmoid(z)
        
        cost = (-1/m) * np.sum(y * np.log(h) + (1-y) * np.log(1-h))
        return float(cost)
    
    def compute_gradients(self, X: ArrayType, y: ArrayType) -> Tuple[ArrayType, float]:
        """Compute gradients for gradient descent.
        
        Args:
            X: Training features of shape (m, n).
            y: Target values of shape (m,).
            
        Returns:
            Tuple containing:
                dj_dw: Gradients with respect to weights.
                dj_db: Gradient with respect to bias.
                
        Raises:
            ValueError: If model is not fitted or input shapes don't match.
        """
        if self.weights is None or self.bias is None:
            raise ValueError("Model must be fitted before computing gradients")
            
        m = X.shape[0]
        z = np.dot(X, self.weights) + self.bias
        h = self.sigmoid(z)
        
        dj_dw = (1/m) * np.dot(X.T, (h - y))
        dj_db = (1/m) * np.sum(h - y)
        
        return dj_dw, dj_db
    
    def fit(self, X: ArrayType, y: ArrayType) -> None:
        """Fit the logistic regression model using gradient descent.
        
        Args:
            X: Training features of shape (m, n).
            y: Target values of shape (m,).
            
        Raises:
            ValueError: If input shapes don't match.
        """
        if X.shape[0] != y.shape[0]:
            raise ValueError("Number of training examples must match")
            
        # Initialize parameters
        n_features = X.shape[1]
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.cost_history = []
        
        # Gradient descent
        for i in range(self.n_iterations):
            # Compute gradients
            dj_dw, dj_db = self.compute_gradients(X, y)
            
            # Update parameters
            self.weights -= self.learning_rate * dj_dw
            self.bias -= self.learning_rate * dj_db
            
            # Save cost
            if i < 100000:  # Prevent resource exhaustion
                cost = self.compute_cost(X, y)
                self.cost_history.append(cost)
    
    def predict_proba(self, X: ArrayType) -> ArrayType:
        """Predict class probabilities.
        
        Args:
            X: Input features of shape (m, n).
            
        Returns:
            Predicted probabilities of shape (m,).
            
        Raises:
            ValueError: If model is not fitted.
        """
        if self.weights is None or self.bias is None:
            raise ValueError("Model must be fitted before making predictions")
            
        z = np.dot(X, self.weights) + self.bias
        return self.sigmoid(z)
    
    def predict(self, X: ArrayType, threshold: float = 0.5) -> ArrayType:
        """Predict class labels.
        
        Args:
            X: Input features of shape (m, n).
            threshold: Classification threshold.
            
        Returns:
            Predicted labels of shape (m,).
            
        Raises:
            ValueError: If model is not fitted.
        """
        probas = self.predict_proba(X)
        return (probas >= threshold).astype(int) 