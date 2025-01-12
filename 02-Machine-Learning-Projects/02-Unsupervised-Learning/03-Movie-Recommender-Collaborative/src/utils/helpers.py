"""
Utility functions for data loading, processing and visualization.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Optional
import numpy.typing as npt

ArrayType = npt.NDArray[np.float64]

def load_ratings(filepath: str) -> Tuple[ArrayType, ArrayType]:
    """Load movie ratings data.
    
    Args:
        filepath: Path to the ratings data file.
        
    Returns:
        Tuple containing:
            Y: Rating matrix (num_movies, num_users).
            R: Binary-valued indicator matrix (num_movies, num_users).
            
    Raises:
        FileNotFoundError: If data file doesn't exist.
    """
    try:
        data = np.load(filepath)
        return data['Y'], data['R']
    except FileNotFoundError:
        raise FileNotFoundError(f"Ratings data file not found at {filepath}")

def load_movie_list(filepath: str) -> Tuple[List[str], pd.DataFrame]:
    """Load movie list and metadata.
    
    Args:
        filepath: Path to the movie list CSV file.
        
    Returns:
        Tuple containing:
            List of movie titles.
            DataFrame containing movie metadata.
            
    Raises:
        FileNotFoundError: If movie list file doesn't exist.
    """
    try:
        df = pd.read_csv(filepath)
        return df['title'].tolist(), df
    except FileNotFoundError:
        raise FileNotFoundError(f"Movie list file not found at {filepath}")

def plot_cost_history(history: Dict[str, list], title: str = "Training History") -> None:
    """Plot training cost history.
    
    Args:
        history: Dictionary containing training history.
        title: Plot title.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history['cost'])
    plt.title(title)
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.grid(True)
    plt.show()

def plot_rating_distribution(Y: ArrayType, R: ArrayType,
                           title: str = "Rating Distribution") -> None:
    """Plot distribution of movie ratings.
    
    Args:
        Y: Rating matrix (num_movies, num_users).
        R: Binary-valued indicator matrix (num_movies, num_users).
        title: Plot title.
    """
    ratings = Y[R > 0]  # Get only rated values
    
    plt.figure(figsize=(10, 6))
    plt.hist(ratings, bins=20, edgecolor='black')
    plt.title(title)
    plt.xlabel('Rating')
    plt.ylabel('Count')
    plt.grid(True)
    plt.show()

def compute_rating_stats(Y: ArrayType, R: ArrayType) -> Dict[str, float]:
    """Compute statistics about movie ratings.
    
    Args:
        Y: Rating matrix (num_movies, num_users).
        R: Binary-valued indicator matrix (num_movies, num_users).
        
    Returns:
        Dictionary containing rating statistics.
    """
    ratings = Y[R > 0]
    
    stats = {
        'mean': float(np.mean(ratings)),
        'std': float(np.std(ratings)),
        'min': float(np.min(ratings)),
        'max': float(np.max(ratings)),
        'count': int(np.sum(R)),
        'sparsity': float(1 - np.sum(R) / (Y.shape[0] * Y.shape[1]))
    }
    
    return stats

def format_movie_recommendations(movie_indices: ArrayType,
                               predicted_ratings: ArrayType,
                               movie_list: List[str]) -> pd.DataFrame:
    """Format movie recommendations into a DataFrame.
    
    Args:
        movie_indices: Indices of recommended movies.
        predicted_ratings: Predicted ratings for recommended movies.
        movie_list: List of movie titles.
        
    Returns:
        DataFrame containing recommendations.
    """
    recommendations = pd.DataFrame({
        'Movie': [movie_list[i] for i in movie_indices],
        'Predicted Rating': predicted_ratings
    })
    
    return recommendations.sort_values('Predicted Rating', ascending=False)

def save_model_parameters(X: ArrayType, W: ArrayType, b: ArrayType,
                        filepath: str) -> None:
    """Save model parameters to file.
    
    Args:
        X: Movie feature matrix.
        W: User parameter matrix.
        b: User bias vector.
        filepath: Path to save parameters.
    """
    np.savez(filepath, X=X, W=W, b=b)

def load_model_parameters(filepath: str
                        ) -> Tuple[ArrayType, ArrayType, ArrayType]:
    """Load model parameters from file.
    
    Args:
        filepath: Path to parameter file.
        
    Returns:
        Tuple containing:
            X: Movie feature matrix.
            W: User parameter matrix.
            b: User bias vector.
            
    Raises:
        FileNotFoundError: If parameter file doesn't exist.
    """
    try:
        data = np.load(filepath)
        return data['X'], data['W'], data['b']
    except FileNotFoundError:
        raise FileNotFoundError(f"Parameter file not found at {filepath}") 