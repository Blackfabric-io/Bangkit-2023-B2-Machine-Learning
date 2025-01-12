"""
Helper functions for data handling and visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional
import numpy.typing as npt
from sklearn.datasets import make_regression
import pandas as pd

MatrixType = npt.NDArray[np.float64]

def generate_regression_data(m: int = 30, n_features: int = 1, 
                           noise: float = 20, random_state: int = 1) -> Tuple[MatrixType, MatrixType]:
    """Generate synthetic regression data.
    like creating a practice game for your team! 🏀📊

    This function creates synthetic data (fake data) that you can use to practice 
    regression models. It's like setting up a scrimmage for your team to practice 
    before the real game!

    Args:
        m: 
            Number of samples (how many players are on the field).
        n_features: 
            Number of features (how many stats you're tracking for each player).
        noise: 
            Standard deviation of Gaussian noise (how "messy" the data is
            like random mistakes during practice).
        random_state: 
            Random seed for reproducibility 
            (so you can replay the same practice game exactly the same way).

    Returns:
        A tuple containing:
            X: Input features of shape (n_features, m) (your players' stats).
            Y: Target values of shape (1, m) (the score you're trying to predict).

    Tip: Use this to test regression models before applying to real data
    like practicing before the big game! 🏆✨
    """
    if m <= 0 or n_features <= 0:
        raise ValueError("m and n_features must be positive")
        
    X, Y = make_regression(n_samples=m, n_features=n_features, 
                          noise=noise, random_state=random_state)
    
    X = X.reshape((n_features, m))
    Y = Y.reshape((1, m))
    
    return X, Y

def load_house_prices_data(filepath: str) -> Tuple[MatrixType, MatrixType]:
    """Load and preprocess house prices dataset.
    🏠 Load and clean up house price data
    like tidying up your room before guests arrive 🧹✨

    This function loads a dataset of house prices from a CSV file 
    preprocesses it to make it ready for analysis. 
    like cleaning up your room so everything is neat and easy to find!

    Args:
        filepath: Path to CSV file (where your messy data lives).

    Returns:
        A tuple containing:
            X: Features (GrLivArea, OverallQual) of shape (2, m) (the size and quality 
               of the houses).
            Y: Target values (SalePrice) of shape (1, m) (the price of the houses—like 
               how much someone would pay).

    Tip: This function helps you focus on the most important features (size and 
    quality) to predict house prices—like picking the best parts to show 
    off to guests!
    """
    if not isinstance(filepath, str):
        raise TypeError("filepath must be a string")
        
    try:
        df = pd.read_csv(filepath)
        X = df[['GrLivArea', 'OverallQual']].values.T
        Y = df['SalePrice'].values.reshape(1, -1)
        return X, Y
    except Exception as e:
        raise RuntimeError(f"Error loading data: {str(e)}")

def plot_regression_line(X: MatrixType, Y: MatrixType, 
                        parameters: Dict[str, MatrixType], 
                        X_pred: Optional[MatrixType] = None) -> None:
    """Plot data points, regression line and predictions.
    Plot your data, the regression line, and predictions
    like drawing a map to show where your team is going! 

    This function helps you visualize your data, the regression line (the best-fit line), 
    and any predictions your model makes. It's like drawing a map to see how well your 
    team is performing and where they're headed!

    This function helps you visualize house price data, the regression line (the best-fit line), 
    and any predictions your model makes. 
    It's like drawing a map to see how house prices change based on size and quality—and guessing what future houses might cost!

    Args:
        X: 
            Input features (your team's stats—like points scored or distance run).
        Y: 
            True labels (the actual results—like whether your team won or lost).
        parameters: 
            Model parameters (the strategy your team is using to win).
        X_pred: 
            Points to make predictions for (optional—like guessing where 
            your team will be in the future).

    Args:
        X: 
            Input features (house size and quality—like how big and nice the house is).
        Y: 
            True labels (actual house prices—like how much the house sold for).
        parameters: 
            Model parameters (the "rules" your model uses to predict prices).
        X_pred: 
            Points to make predictions for (optional—like guessing the price of 
            a new house based on its size and quality).

    Tip: Use this to see how well your regression model fits the data
    like checking if your team's strategy is working!

    Use this to see how well your regression model fits the data—like checking 
    if your "house price predictor" is accurate! 🏠💰
    """
    if not isinstance(X, np.ndarray) or not isinstance(Y, np.ndarray):
        raise TypeError("X and Y must be numpy arrays")
    if X.shape[0] != 1:
        raise ValueError("This function only works for 1D regression")
        
    W = parameters["W"]
    b = parameters["b"]
    
    plt.scatter(X[0,:], Y[0,:], color="black")
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    
    X_line = np.arange(np.min(X[0,:]), np.max(X[0,:])*1.1, 0.1)
    plt.plot(X_line, W[0,0] * X_line + b[0,0], "r")
    
    if X_pred is not None:
        Y_pred = W[0,0] * X_pred + b[0,0]
        plt.plot(X_pred, Y_pred, "bo")
    
    plt.show() 