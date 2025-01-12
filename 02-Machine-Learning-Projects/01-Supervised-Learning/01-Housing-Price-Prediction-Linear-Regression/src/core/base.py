"""
Core functionality for linear regression model.
"""

import numpy as np
from typing import Tuple, Optional
import numpy.typing as npt

ArrayType = npt.NDArray[np.float64]

class LinearRegression:
    """Linear regression model implementation.
    
    This class implements linear regression using gradient descent optimization.
    The model fits a line f(x) = wx + b to the data by minimizing the mean squared error.
    """
    
    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 1500):
        """Initialize the model.
        
        Args:
            learning_rate: Learning rate for gradient descent.
            n_iterations: Number of gradient descent iterations.
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.w: Optional[float] = None
        self.b: Optional[float] = None
        self.cost_history: list = []
        
    def compute_cost(self, X: ArrayType, y: ArrayType) -> float:
        """Compute the mean squared error cost.
        
        Args:
            X: Training features of shape (m,).
            y: Target values of shape (m,).
            
        Returns:
            Cost value.
            
        Raises:
            ValueError: If model is not fitted or input shapes don't match.
        """
        if self.w is None or self.b is None:
            raise ValueError("Model must be fitted before computing cost")
            
        m = X.shape[0]
        cost = 0.0
        
        for i in range(m):
            f_wb = self.w * X[i] + self.b
            cost += (f_wb - y[i]) ** 2
            
        return cost / (2 * m)
    
    def compute_gradient(self, X: ArrayType, y: ArrayType) -> Tuple[float, float]:
        """Compute gradients for gradient descent.
        
        Args:
            X: Training features of shape (m,).
            y: Target values of shape (m,).
            
        Returns:
            Tuple containing:
                dj_dw: Gradient with respect to w.
                dj_db: Gradient with respect to b.
                
        Raises:
            ValueError: If model is not fitted or input shapes don't match.
        """
        if self.w is None or self.b is None:
            raise ValueError("Model must be fitted before computing gradient")
            
        m = X.shape[0]
        dj_dw = 0
        dj_db = 0
        
        for i in range(m):
            f_wb = self.w * X[i] + self.b
            dj_dw += (f_wb - y[i]) * X[i]
            dj_db += f_wb - y[i]
            
        dj_dw /= m
        dj_db /= m
        
        return dj_dw, dj_db
    
    def fit(self, X: ArrayType, y: ArrayType) -> None:
        """Fit the linear regression model using gradient descent.
        
        Args:
            X: Training features of shape (m,).
            y: Target values of shape (m,).
            
        Raises:
            ValueError: If input shapes don't match.
        """
        if X.shape[0] != y.shape[0]:
            raise ValueError("Number of training examples must match")
            
        # Initialize parameters
        self.w = 0.0
        self.b = 0.0
        self.cost_history = []
        
        # Gradient descent
        for i in range(self.n_iterations):
            # Compute gradients
            dj_dw, dj_db = self.compute_gradient(X, y)
            
            # Update parameters
            self.w -= self.learning_rate * dj_dw
            self.b -= self.learning_rate * dj_db
            
            # Save cost
            if i < 100000:  # Prevent resource exhaustion
                cost = self.compute_cost(X, y)
                self.cost_history.append(cost)
    
    def predict(self, X: ArrayType) -> ArrayType:
        """Make predictions using the fitted model.
        
        Args:
            X: Input features of shape (m,).
            
        Returns:
            Predicted values of shape (m,).
            
        Raises:
            ValueError: If model is not fitted.
        """
        if self.w is None or self.b is None:
            raise ValueError("Model must be fitted before making predictions")
            
        return self.w * X + self.b 