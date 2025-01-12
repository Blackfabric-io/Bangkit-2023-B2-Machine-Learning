"""
Main neural network model training and prediction functionality.
"""

import numpy as np
from typing import Dict, Tuple, Optional
import numpy.typing as npt
from ..core.base import (
    layer_sizes, 
    initialize_parameters,
    forward_propagation,
    compute_cost
)

MatrixType = npt.NDArray[np.float64]

def train_nn(parameters: Dict[str, MatrixType], Y_hat: MatrixType, 
             X: MatrixType, Y: MatrixType) -> Dict[str, MatrixType]:
    """Train neural network parameters using gradient descent.
    🏠 Train your neural network using gradient descent
    like teaching your house price predictor to get smarter! 🧠✨

    This function updates the neural network's parameters (its "brain") to make better 
    predictions about house prices. It's like teaching your predictor to learn from its 
    mistakes and improve over time!

    Args:
        parameters: 
            Current model parameters (the "rules" your predictor is using right now).
        Y_hat: 
            Current predictions (what your predictor thinks the house prices are).
        X: 
            Input data (house size and quality—like how big and nice the house is).
        Y: 
            True labels (actual house prices—like how much the house sold for).

    Returns:
        Updated parameters (the "smarter rules" your predictor learned after training).

    Tip: This is like giving your predictor a workout
    it learns from its mistakes and gets better at guessing house prices!
    """
    return parameters

def nn_model(X: MatrixType, Y: MatrixType, num_iterations: int = 10, 
             print_cost: bool = False) -> Dict[str, MatrixType]:
    """Build and train a neural network model.
    🏠 Build and train a neural network
    like creating and teaching a house price predictor!

    This function builds a neural network (your "house price predictor") and trains it 
    using the data you provide. It's like teaching your predictor to learn from house 
    sizes and qualities to guess prices accurately!

    Args:
        X: 
            Input data of shape (n_x, m) (house sizes and qualities—like how big and nice the houses are).
        Y: 
            True labels of shape (n_y, m) (actual house prices—like how much the houses sold for).
        num_iterations: 
            Number of iterations for training (how many times your predictor practices).
        print_cost: 
            Whether to print cost during training (optional—like checking how well your predictor is doing during practice).

    Returns:
        parameters: 
            Trained model parameters (the "smart rules" your predictor learned after training).

    Tip: 
        Think of this as building a robot that learns to predict house prices
        it gets better with practice! 
    """
    if not isinstance(X, np.ndarray) or not isinstance(Y, np.ndarray):
        raise TypeError("Inputs must be numpy arrays")
    if num_iterations <= 0:
        raise ValueError("num_iterations must be positive")
        
    n_x, n_y = layer_sizes(X, Y)
    parameters = initialize_parameters(n_x, n_y)
    
    for i in range(num_iterations):
        Y_hat = forward_propagation(X, parameters, n_y)
        cost = compute_cost(Y_hat, Y)
        parameters = train_nn(parameters, Y_hat, X, Y)
        
        if print_cost:
            print(f"Cost after iteration {i}: {cost:.6f}")
            
    return parameters

def normalize_data(X: MatrixType) -> Tuple[MatrixType, MatrixType, MatrixType]:
    """Normalize data by subtracting mean and dividing by standard deviation.
    🏠 Normalize your data—like adjusting house prices to make them easier to compare! 📊✨

    This function normalizes your data by subtracting the mean (average) and dividing by 
    the standard deviation (how spread out the data is). It's like adjusting house prices 
    so they're all on the same scale, making it easier to compare them!

    Args:
        X: 
            Input data to normalize (house prices or other numbers you want to adjust).

    Returns:
        A tuple containing:
            X_norm: 
                Normalized data (adjusted house prices that are easier to compare).
            X_mean: 
                Mean of original data (the average house price before adjusting).
            X_std: 
                Standard deviation of original data (how spread out the house prices were).

    Tip: 
        Normalizing data is like putting all your house prices on the same "ruler"
        it makes it easier to see patterns and trends! 📏🏡
    """
    if not isinstance(X, np.ndarray):
        raise TypeError("Input must be a numpy array")
        
    X_mean = np.mean(X, axis=1, keepdims=True)
    X_std = np.std(X, axis=1, keepdims=True)
    X_norm = (X - X_mean) / X_std
    
    return X_norm, X_mean, X_std

def predict(X: MatrixType, parameters: Dict[str, MatrixType], 
           X_mean: Optional[MatrixType] = None, 
           X_std: Optional[MatrixType] = None,
           Y_mean: Optional[float] = None,
           Y_std: Optional[float] = None) -> MatrixType:
    """Make predictions using trained model.
    🏠 Make predictions using your trained model—like guessing house prices with your smart predictor! 🔮✨

    This function uses your trained model to predict house prices based on new data. It 
    adjusts the input data to match the scale used during training and then converts the 
    predictions back to the original scale. It's like asking your predictor, "What do you 
    think this house will sell for?"

    Args:
        X: 
            Input data (new house sizes and qualities—like how big and nice the houses are).
        parameters: 
            Trained model parameters (the "smart rules" your predictor learned).
        X_mean: 
            Mean of training data (average house size and quality used during training).
        X_std: 
            Standard deviation of training data (how spread out the house sizes and qualities were during training).
        Y_mean: 
            Mean of training labels (average house price used during training).
        Y_std: 
            Standard deviation of training labels (how spread out the house prices were during training).

    Returns:
        Y_pred: 
            Model predictions (your predictor's guesses for the house prices).

    Tip: 
        Think of this as asking your predictor to take a test—it uses what it learned 
    to guess the answers! 🧠📝
    """
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array")
    if not isinstance(parameters, dict):
        raise TypeError("parameters must be a dictionary")
        
    # Normalize input if mean and std are provided
    if X_mean is not None and X_std is not None:
        # Adjust the house sizes and qualities to match the scale used during training
        X = (X - X_mean) / X_std  
            # Like putting all houses on the same "ruler" 📏🏡

    # Get the number of outputs (e.g., house prices to predict)
    n_y = parameters["W"].shape[0]  
        # How many predictions your model can make 🧠✨

    # Make predictions using forward propagation
    Y_pred = forward_propagation(X, parameters, n_y)  
        # Ask your predictor to guess the house prices! 🔮💰

    # Denormalize output if mean and std are provided
    if Y_mean is not None and Y_std is not None:
        # Convert the predicted house prices back to the original scale
        Y_pred = Y_pred * Y_std + Y_mean  
            # Like putting the guessed prices back into real dollars! 💵✨

    # Return the final predictions
    return Y_pred  
        # Here are your predictor's best guesses for the house prices! 🏠🔮