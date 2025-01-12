"""
Core neural network operations and model architecture.
"""

import numpy as np
from typing import Dict, Tuple
import numpy.typing as npt

MatrixType = npt.NDArray[np.float64]

def layer_sizes(X: MatrixType, Y: MatrixType) -> Tuple[int, int]:
    """Get the size of input and output layers.
    Figure out how big your neural network's "brain" needs to be! 🧠✨

    This function calculates the size of the input and output layers for your neural network. 
    Think of it like figuring out how many questions your predictor needs to ask about a house 
    (input size) and how many answers it needs to give (output size).

    Args:
        X: 
            Input dataset of shape (input size, number of examples) 
            (e.g., house sizes and qualities for each house in your dataset).
        Y: 
            Labels of shape (output size, number of examples)
            (e.g., actual house prices for each house in your dataset).

    Returns:
        A tuple containing:
            n_x: 
                Size of the input layer (how many questions your predictor needs to ask about each house).
            n_y: 
                Size of the output layer (how many answers your predictor needs to give, like the predicted price).

    Tip: 
        This is like setting up the "brain" of your house price predictor—
        it needs to know how much information to take in and how much to spit out! 🏡💡
    """
    if not isinstance(X, np.ndarray) or not isinstance(Y, np.ndarray):
        raise TypeError("Inputs must be numpy arrays")
        
    n_x = X.shape[0]
    n_y = Y.shape[0]
    
    return n_x, n_y

def initialize_parameters(n_x: int, n_y: int) -> Dict[str, MatrixType]:
    """Initialize neural network parameters.
    🏠 Set up the "brain" of your house price predictor! 🧠✨

    This function creates the starting "rules" (weights and biases) for your neural network. 
    Think of it like giving your predictor a blank notebook to write down its guesses about 
    house prices. The weights and biases are like the first scribbles it makes before it 
    learns anything!

    Args:
        n_x: 
            Size of the input layer (how many questions your predictor needs to ask about each house).
        n_y: 
            Size of the output layer (how many answers your predictor needs to give, like the predicted price).

    Returns:
        A dictionary containing:
            W: 
                Weight matrix of shape (n_y, n_x) (the "rules" for how much each question matters).
            b: 
                Bias vector of shape (n_y, 1) (the "starting point" for your predictor's guesses).

    Tip: This is like giving your predictor a fresh notebook and saying, "Start guessing!" 
    It will get better as it learns! 📓🔮
    """
    if n_x <= 0 or n_y <= 0:
        raise ValueError("Layer sizes must be positive")
        
    W = np.random.randn(n_y, n_x) * 0.01
    b = np.zeros((n_y, 1))
    
    parameters = {
        "W": W,
        "b": b
    }
    
    return parameters

def forward_propagation(X: MatrixType, parameters: Dict[str, MatrixType], n_y: int) -> MatrixType:
    """Implement forward propagation
    🏠 Let your house price predictor make its first guesses! 🔮✨

    This function takes the input data (like house sizes and qualities) and uses the "rules" 
    (weights and biases) to make predictions. Think of it like your predictor looking at a 
    house, using its notebook of rules, and saying, "I think this house is worth this much!"

    Args:
        X: 
            Input data of shape (n_x, m) (e.g., house sizes and qualities for each house in your dataset).
        parameters: 
            Dictionary containing W (weights) and b (bias) (the "rules" your predictor uses to make guesses).
        n_y: 
            Size of the output layer (how many answers your predictor needs to give, like the predicted price).

    Returns:
        Y_hat: 
            The output predictions of shape (n_y, m) (your predictor's guesses for the house prices).

    Tip: 
        This is like your predictor taking a test—it uses its rules to guess the answers, 
    but it might not be perfect yet! 🧠📝
    """
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array (like a list of house data!)")
    if not isinstance(parameters, dict):
        raise TypeError("parameters must be a dictionary (like a notebook of rules!)")
    if "W" not in parameters or "b" not in parameters:
        raise KeyError("parameters must contain 'W' and 'b' keys (your predictor needs its rules!)")
        
    W = parameters["W"]  # The "rules" for how much each question matters
    b = parameters["b"]  # The "starting point" for your predictor's guesses
    
    Z = np.dot(W, X) + b  # Combine the rules and data to make a guess
    Y_hat = Z  # This is your predictor's guess for the house prices!
    
    assert(Y_hat.shape == (n_y, X.shape[1]))  # Make sure the guesses are the right shape
    
    return Y_hat  # Here are your predictor's guesses! 🏡💡


def compute_cost(Y_hat: MatrixType, Y: MatrixType) -> float:
    """Compute the cost function as mean squared error
    🏠 Check how good your house price predictor's guesses are! 📊✨

    This function calculates how far off your predictor's guesses (Y_hat) are from the actual 
    house prices (Y). Think of it like grading a test—the lower the score, the better your 
    predictor is doing!

    Args:
        Y_hat: 
            Model predictions of shape (n_y, m) (your predictor's guesses for house prices).
        Y: 
            True labels of shape (n_y, m) (the actual house prices).

    Returns:
        cost: 
            A number that tells you how wrong your predictor is (the lower, the better!).

    Tip: 
        This is like a "mistake meter" for your predictor
        it helps it learn and get better! 📉🧠
    """
    if not isinstance(Y_hat, np.ndarray) or not isinstance(Y, np.ndarray):
        raise TypeError("Inputs must be numpy arrays (like lists of numbers!)")
    if Y_hat.shape != Y.shape:
        raise ValueError("Y_hat and Y must have the same shape (your guesses and the real prices need to match!)")
        
    m = Y.shape[1]  # Number of houses in your dataset
    cost = np.sum((Y_hat - Y)**2)/(2*m)  # Calculate how far off your guesses are
    
    return cost  # Here's the "mistake meter" score! 🏡📉
