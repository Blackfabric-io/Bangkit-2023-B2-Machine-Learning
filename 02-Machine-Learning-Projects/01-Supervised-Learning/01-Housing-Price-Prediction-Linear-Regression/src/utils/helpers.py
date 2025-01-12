"""
Utility functions for data preprocessing and visualization.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple
import numpy.typing as npt
from sklearn.preprocessing import StandardScaler

ArrayType = npt.NDArray[np.float64]

def load_data(filepath: str) -> pd.DataFrame:
    """Load housing data from CSV file.
    
    Args:
        filepath: Path to the CSV file.
        
    Returns:
        DataFrame containing housing data.
        
    Raises:
        FileNotFoundError: If file doesn't exist.
    """
    try:
        return pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found at {filepath}")

def preprocess_data(df: pd.DataFrame, feature_col: str, target_col: str) -> Tuple[ArrayType, ArrayType]:
    """Preprocess data for model training.
    
    Args:
        df: Input DataFrame.
        feature_col: Name of feature column.
        target_col: Name of target column.
        
    Returns:
        Tuple containing:
            X: Scaled feature array.
            y: Target array.
    """
    X = df[feature_col].values
    y = df[target_col].values
    
    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X.reshape(-1, 1)).ravel()
    
    return X, y

def plot_data(X: ArrayType, y: ArrayType, title: str = "Housing Data") -> None:
    """Plot training data.
    
    Args:
        X: Feature array.
        y: Target array.
        title: Plot title.
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='blue', alpha=0.5)
    plt.xlabel('Scaled Square Footage')
    plt.ylabel('Price')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_regression_line(X: ArrayType, y: ArrayType, w: float, b: float, 
                        title: str = "Linear Regression Fit") -> None:
    """Plot data points and regression line.
    
    Args:
        X: Feature array.
        y: Target array.
        w: Model weight.
        b: Model bias.
        title: Plot title.
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='blue', alpha=0.5, label='Data points')
    
    # Generate points for regression line
    x_line = np.linspace(X.min(), X.max(), 100)
    y_line = w * x_line + b
    
    plt.plot(x_line, y_line, color='red', label='Regression line')
    plt.xlabel('Scaled Square Footage')
    plt.ylabel('Price')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_cost_history(cost_history: list, title: str = "Cost History") -> None:
    """Plot training cost history.
    
    Args:
        cost_history: List of cost values during training.
        title: Plot title.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(cost_history)
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.show()

def compute_metrics(y_true: ArrayType, y_pred: ArrayType) -> dict:
    """Compute regression metrics.
    
    Args:
        y_true: True target values.
        y_pred: Predicted target values.
        
    Returns:
        Dictionary containing MSE, R2 score, and MAPE.
    """
    mse = np.mean((y_true - y_pred) ** 2)
    r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    return {
        'mse': mse,
        'r2_score': r2,
        'mape': mape
    } 