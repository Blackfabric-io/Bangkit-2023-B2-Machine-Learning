"""
Main processing logic for movie recommender system.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from pathlib import Path
import numpy.typing as npt
from src.core.base import CollaborativeFilter
from src.utils.helpers import (
    load_ratings,
    load_movie_list,
    plot_cost_history,
    plot_rating_distribution,
    compute_rating_stats,
    format_movie_recommendations,
    save_model_parameters,
    load_model_parameters
)

ArrayType = npt.NDArray[np.float64]

class MovieRecommender:
    """Movie recommender system using collaborative filtering."""
    
    def __init__(self, num_features: int = 100, learning_rate: float = 0.1,
                 num_iterations: int = 200, lambda_: float = 1.0):
        """Initialize the movie recommender.
        
        Args:
            num_features: Number of latent features.
            learning_rate: Learning rate for gradient descent.
            num_iterations: Number of training iterations.
            lambda_: Regularization parameter.
        """
        self.model = CollaborativeFilter(
            num_features=num_features,
            learning_rate=learning_rate,
            num_iterations=num_iterations,
            lambda_=lambda_
        )
        self.movie_list: Optional[List[str]] = None
        self.movie_data: Optional[pd.DataFrame] = None
        
    def load_data(self, ratings_path: str, movie_list_path: str) -> Dict[str, Any]:
        """Load ratings and movie data.
        
        Args:
            ratings_path: Path to ratings data file.
            movie_list_path: Path to movie list file.
            
        Returns:
            Dictionary containing data statistics.
        """
        # Load data
        Y, R = load_ratings(ratings_path)
        self.movie_list, self.movie_data = load_movie_list(movie_list_path)
        
        # Compute and display statistics
        stats = compute_rating_stats(Y, R)
        print("\nDataset Statistics:")
        print(f"Number of movies: {Y.shape[0]}")
        print(f"Number of users: {Y.shape[1]}")
        print(f"Number of ratings: {stats['count']}")
        print(f"Rating range: [{stats['min']:.1f}, {stats['max']:.1f}]")
        print(f"Mean rating: {stats['mean']:.2f} ± {stats['std']:.2f}")
        print(f"Matrix sparsity: {stats['sparsity']*100:.1f}%")
        
        # Plot rating distribution
        plot_rating_distribution(Y, R)
        
        return stats
    
    def train(self, ratings_path: str) -> Dict[str, list]:
        """Train the recommender system.
        
        Args:
            ratings_path: Path to ratings data file.
            
        Returns:
            Dictionary containing training history.
            
        Raises:
            FileNotFoundError: If data file doesn't exist.
        """
        # Load ratings
        Y, R = load_ratings(ratings_path)
        
        # Train model
        print("\nTraining collaborative filtering model...")
        history = self.model.fit(Y, R)
        
        # Plot training history
        plot_cost_history(history)
        
        return history
    
    def recommend_for_user(self, user_ratings: ArrayType,
                          n_recommendations: int = 10) -> pd.DataFrame:
        """Generate movie recommendations for a user.
        
        Args:
            user_ratings: Vector of user ratings.
            n_recommendations: Number of recommendations to generate.
            
        Returns:
            DataFrame containing recommendations.
            
        Raises:
            ValueError: If movie_list is not loaded or user_ratings length is invalid.
        """
        if self.movie_list is None:
            raise ValueError("Movie list must be loaded before making recommendations")
        if len(user_ratings) != len(self.movie_list):
            raise ValueError("user_ratings length must match number of movies")
            
        # Create rating matrices for user
        Y = np.c_[user_ratings, np.zeros((len(user_ratings), self.model.W.shape[0]-1))]
        R = np.c_[(user_ratings != 0).astype(int),
                  np.zeros((len(user_ratings), self.model.W.shape[0]-1))]
        
        # Get recommendations
        movie_indices, predicted_ratings = self.model.recommend_movies(
            user_idx=0,  # New user is at index 0
            n_recommendations=n_recommendations,
            exclude_rated=True,
            R=R
        )
        
        # Format recommendations
        return format_movie_recommendations(
            movie_indices, predicted_ratings, self.movie_list
        )
    
    def save_model(self, model_dir: str) -> None:
        """Save trained model and data.
        
        Args:
            model_dir: Directory to save model files.
            
        Raises:
            ValueError: If model is not trained.
        """
        if self.model.X is None:
            raise ValueError("Model must be trained before saving")
            
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model parameters
        save_model_parameters(
            self.model.X.numpy(),
            self.model.W.numpy(),
            self.model.b.numpy(),
            model_dir / 'model_params.npz'
        )
        
        # Save movie list if available
        if self.movie_data is not None:
            self.movie_data.to_csv(model_dir / 'movies.csv', index=False)
    
    @classmethod
    def load_model(cls, model_dir: str,
                  hyperparameters: Optional[Dict] = None) -> 'MovieRecommender':
        """Load trained model and data.
        
        Args:
            model_dir: Directory containing model files.
            hyperparameters: Optional dictionary of model hyperparameters.
            
        Returns:
            Loaded MovieRecommender instance.
            
        Raises:
            FileNotFoundError: If model files don't exist.
        """
        model_dir = Path(model_dir)
        
        # Create instance with optional hyperparameters
        instance = cls(**(hyperparameters or {}))
        
        # Load model parameters
        X, W, b = load_model_parameters(model_dir / 'model_params.npz')
        instance.model.X = tf.Variable(X, name='X')
        instance.model.W = tf.Variable(W, name='W')
        instance.model.b = tf.Variable(b, name='b')
        
        # Load movie data if available
        movie_file = model_dir / 'movies.csv'
        if movie_file.exists():
            instance.movie_list, instance.movie_data = load_movie_list(str(movie_file))
        
        return instance 