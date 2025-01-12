"""
Core implementation of collaborative filtering recommender system.
"""

import numpy as np
import tensorflow as tf
from typing import Tuple, Optional, Dict
import numpy.typing as npt

ArrayType = npt.NDArray[np.float64]

class CollaborativeFilter:
    """Collaborative filtering recommender system using matrix factorization."""
    
    def __init__(self, num_features: int = 100, learning_rate: float = 0.1,
                 num_iterations: int = 200, lambda_: float = 1.0):
        """Initialize the collaborative filter.
        
        Args:
            num_features: Number of latent features.
            learning_rate: Learning rate for gradient descent.
            num_iterations: Number of training iterations.
            lambda_: Regularization parameter.
        """
        self.num_features = num_features
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.lambda_ = lambda_
        
        # Model parameters
        self.X: Optional[tf.Variable] = None  # Movie features
        self.W: Optional[tf.Variable] = None  # User parameters
        self.b: Optional[tf.Variable] = None  # User bias
        self.Y_mean: Optional[ArrayType] = None  # Mean ratings
        
    def _compute_cost(self, X: ArrayType, W: ArrayType, b: ArrayType,
                     Y: ArrayType, R: ArrayType, lambda_: float) -> float:
        """Compute the cost function for collaborative filtering.
        
        Args:
            X: Movie feature matrix (num_movies, num_features).
            W: User parameter matrix (num_users, num_features).
            b: User bias vector (1, num_users).
            Y: Rating matrix (num_movies, num_users).
            R: Binary-valued indicator matrix (num_movies, num_users).
            lambda_: Regularization parameter.
            
        Returns:
            Cost value.
        """
        j = (tf.linalg.matmul(X, tf.transpose(W)) + b - Y) * R
        cost = 0.5 * tf.reduce_sum(j**2)
        cost += (lambda_/2) * (tf.reduce_sum(X**2) + tf.reduce_sum(W**2))
        return float(cost)
    
    def _normalize_ratings(self, Y: ArrayType, R: ArrayType
                         ) -> Tuple[ArrayType, ArrayType]:
        """Normalize ratings by subtracting the mean rating for each movie.
        
        Args:
            Y: Rating matrix (num_movies, num_users).
            R: Binary-valued indicator matrix (num_movies, num_users).
            
        Returns:
            Tuple containing:
                Normalized rating matrix.
                Mean rating vector.
        """
        Y_mean = np.sum(Y * R, axis=1) / (np.sum(R, axis=1) + 1e-12)
        Y_norm = Y - np.outer(Y_mean, np.ones(Y.shape[1]))
        Y_norm = Y_norm * R  # Keep only rated values
        return Y_norm, Y_mean
    
    def fit(self, Y: ArrayType, R: ArrayType) -> Dict[str, list]:
        """Train the collaborative filtering model.
        
        Args:
            Y: Rating matrix (num_movies, num_users).
            R: Binary-valued indicator matrix (num_movies, num_users).
            
        Returns:
            Dictionary containing training history.
        """
        num_movies, num_users = Y.shape
        
        # Normalize ratings
        Y_norm, self.Y_mean = self._normalize_ratings(Y, R)
        
        # Initialize parameters
        tf.random.set_seed(1234)
        self.W = tf.Variable(
            tf.random.normal((num_users, self.num_features),
                           dtype=tf.float64),
            name='W'
        )
        self.X = tf.Variable(
            tf.random.normal((num_movies, self.num_features),
                           dtype=tf.float64),
            name='X'
        )
        self.b = tf.Variable(
            tf.random.normal((1, num_users), dtype=tf.float64),
            name='b'
        )
        
        # Initialize optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        
        # Training history
        history = {'cost': []}
        
        # Training loop
        for iter in range(self.num_iterations):
            with tf.GradientTape() as tape:
                cost = self._compute_cost(
                    self.X, self.W, self.b, Y_norm, R, self.lambda_
                )
            
            # Compute gradients
            grads = tape.gradient(cost, [self.X, self.W, self.b])
            
            # Update parameters
            optimizer.apply_gradients(zip(grads, [self.X, self.W, self.b]))
            
            # Record history
            history['cost'].append(float(cost))
            
            # Log progress
            if iter % 20 == 0:
                print(f"Iteration {iter}: cost = {cost:0.1f}")
        
        return history
    
    def predict(self, user_idx: Optional[int] = None) -> ArrayType:
        """Predict ratings for all movies or a specific user.
        
        Args:
            user_idx: Optional user index. If None, predict for all users.
            
        Returns:
            Predicted ratings.
            
        Raises:
            ValueError: If model is not trained or user_idx is invalid.
        """
        if self.X is None or self.W is None or self.b is None:
            raise ValueError("Model must be trained before making predictions")
            
        # Compute predictions
        X_np = self.X.numpy()
        W_np = self.W.numpy()
        b_np = self.b.numpy()
        
        if user_idx is not None:
            if user_idx >= W_np.shape[0]:
                raise ValueError(f"Invalid user_idx: {user_idx}")
            predictions = (X_np @ W_np[user_idx].T +
                         b_np[0, user_idx] + self.Y_mean)
        else:
            predictions = (X_np @ W_np.T + b_np + 
                         np.outer(self.Y_mean, np.ones(W_np.shape[0])))
        
        return predictions
    
    def recommend_movies(self, user_idx: int, n_recommendations: int = 10,
                        exclude_rated: bool = True, R: Optional[ArrayType] = None
                        ) -> Tuple[ArrayType, ArrayType]:
        """Recommend top N movies for a user.
        
        Args:
            user_idx: User index.
            n_recommendations: Number of recommendations to return.
            exclude_rated: Whether to exclude already rated movies.
            R: Binary-valued indicator matrix. Required if exclude_rated is True.
            
        Returns:
            Tuple containing:
                Indices of recommended movies.
                Predicted ratings for recommended movies.
                
        Raises:
            ValueError: If exclude_rated is True but R is not provided.
        """
        if exclude_rated and R is None:
            raise ValueError("R matrix required when exclude_rated is True")
            
        # Get predictions for user
        predictions = self.predict(user_idx)
        
        # Exclude rated movies if requested
        if exclude_rated:
            predictions[R[:, user_idx] > 0] = -np.inf
            
        # Get top N recommendations
        movie_indices = np.argsort(predictions)[::-1][:n_recommendations]
        return movie_indices, predictions[movie_indices] 