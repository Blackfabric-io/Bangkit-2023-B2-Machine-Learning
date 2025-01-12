"""
Unit tests for movie recommender system.
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path
import tensorflow as tf
from src.core.base import CollaborativeFilter
from src.processors.main import MovieRecommender
from src.utils.helpers import save_model_parameters

def create_synthetic_data(n_movies: int = 10, n_users: int = 8,
                         n_features: int = 3, sparsity: float = 0.3,
                         random_state: int = 42) -> tuple:
    """Create synthetic rating data for testing.
    
    Args:
        n_movies: Number of movies.
        n_users: Number of users.
        n_features: Number of latent features.
        sparsity: Fraction of ratings that are zero.
        random_state: Random seed.
        
    Returns:
        Tuple of (Y, R) matrices and movie list.
    """
    np.random.seed(random_state)
    
    # Generate true parameters
    X = np.random.normal(0, 1, (n_movies, n_features))
    W = np.random.normal(0, 1, (n_users, n_features))
    b = np.random.normal(0, 0.1, (n_movies, 1))
    
    # Generate ratings
    Y = np.dot(X, W.T) + b
    Y = np.clip(Y, 1, 5)
    
    # Add sparsity
    mask = np.random.random((n_movies, n_users)) > sparsity
    Y = Y * mask
    R = (Y > 0).astype(int)
    
    # Create movie list
    movie_list = [f"Movie {i+1}" for i in range(n_movies)]
    
    return Y, R, movie_list

def test_collaborative_filter_initialization():
    """Test initialization of CollaborativeFilter class."""
    model = CollaborativeFilter(num_features=10)
    assert model.num_features == 10
    assert model.learning_rate == 0.1
    assert model.num_iterations == 200
    assert model.lambda_ == 1.0
    assert model.X is None
    assert model.W is None
    assert model.b is None

def test_collaborative_filter_training():
    """Test training of CollaborativeFilter model."""
    # Create synthetic data
    Y, R, _ = create_synthetic_data()
    
    # Train model
    model = CollaborativeFilter(num_features=3, num_iterations=50)
    history = model.fit(Y, R)
    
    # Check results
    assert isinstance(history, dict)
    assert 'cost' in history
    assert len(history['cost']) == 50
    assert history['cost'][-1] < history['cost'][0]
    assert isinstance(model.X, tf.Variable)
    assert isinstance(model.W, tf.Variable)
    assert isinstance(model.b, tf.Variable)
    assert model.X.shape == (Y.shape[0], 3)
    assert model.W.shape == (Y.shape[1], 3)
    assert model.b.shape == (Y.shape[0], 1)

def test_collaborative_filter_prediction():
    """Test prediction with CollaborativeFilter model."""
    # Create synthetic data
    Y, R, _ = create_synthetic_data()
    
    # Train model
    model = CollaborativeFilter(num_features=3, num_iterations=50)
    model.fit(Y, R)
    
    # Test predictions
    predictions = model.predict()
    assert predictions.shape == Y.shape
    assert np.all(predictions >= 1) and np.all(predictions <= 5)

def test_movie_recommender_initialization():
    """Test initialization of MovieRecommender class."""
    recommender = MovieRecommender(num_features=10)
    assert recommender.model.num_features == 10
    assert recommender.movie_list is None
    assert recommender.movie_data is None

def test_movie_recommender_data_loading(tmp_path):
    """Test data loading in MovieRecommender."""
    # Create synthetic data
    Y, R, movie_list = create_synthetic_data()
    
    # Save data to temporary files
    ratings_path = tmp_path / "ratings.npz"
    np.savez(ratings_path, Y=Y, R=R)
    
    movies_df = pd.DataFrame({
        'title': movie_list,
        'genres': ['Action'] * len(movie_list)
    })
    movies_path = tmp_path / "movies.csv"
    movies_df.to_csv(movies_path, index=False)
    
    # Test loading
    recommender = MovieRecommender()
    stats = recommender.load_data(str(ratings_path), str(movies_path))
    
    assert isinstance(stats, dict)
    assert 'mean' in stats
    assert 'std' in stats
    assert 'count' in stats
    assert len(recommender.movie_list) == len(movie_list)

def test_movie_recommender_training(tmp_path):
    """Test training in MovieRecommender."""
    # Create synthetic data
    Y, R, movie_list = create_synthetic_data()
    
    # Save data
    ratings_path = tmp_path / "ratings.npz"
    np.savez(ratings_path, Y=Y, R=R)
    
    # Test training
    recommender = MovieRecommender(num_features=3, num_iterations=50)
    history = recommender.train(str(ratings_path))
    
    assert isinstance(history, dict)
    assert 'cost' in history
    assert len(history['cost']) == 50
    assert history['cost'][-1] < history['cost'][0]

def test_movie_recommender_recommendations():
    """Test recommendation generation in MovieRecommender."""
    # Create synthetic data
    Y, R, movie_list = create_synthetic_data()
    
    # Create and train recommender
    recommender = MovieRecommender(num_features=3, num_iterations=50)
    recommender.model.fit(Y, R)
    recommender.movie_list = movie_list
    
    # Test recommendations
    user_ratings = np.zeros(len(movie_list))
    user_ratings[0] = 5  # Rate first movie
    
    recommendations = recommender.recommend_for_user(user_ratings, n_recommendations=3)
    
    assert isinstance(recommendations, pd.DataFrame)
    assert len(recommendations) == 3
    assert 'Movie' in recommendations.columns
    assert 'Predicted Rating' in recommendations.columns
    assert recommendations['Predicted Rating'].iloc[0] >= recommendations['Predicted Rating'].iloc[-1]

def test_movie_recommender_save_load(tmp_path):
    """Test model saving and loading in MovieRecommender."""
    # Create synthetic data
    Y, R, movie_list = create_synthetic_data()
    
    # Create and train recommender
    recommender = MovieRecommender(num_features=3, num_iterations=50)
    recommender.model.fit(Y, R)
    recommender.movie_list = movie_list
    
    # Save model
    model_dir = tmp_path / "model"
    recommender.save_model(str(model_dir))
    
    # Load model
    loaded = MovieRecommender.load_model(str(model_dir))
    
    # Compare predictions
    user_ratings = np.zeros(len(movie_list))
    user_ratings[0] = 5
    
    original_recs = recommender.recommend_for_user(user_ratings)
    loaded_recs = loaded.recommend_for_user(user_ratings)
    
    pd.testing.assert_frame_equal(original_recs, loaded_recs)

def test_error_handling():
    """Test error handling in MovieRecommender."""
    recommender = MovieRecommender()
    
    # Test recommendation without movie list
    with pytest.raises(ValueError, match="Movie list must be loaded"):
        recommender.recommend_for_user(np.array([]))
    
    # Test saving untrained model
    with pytest.raises(ValueError, match="Model must be trained"):
        recommender.save_model("dummy_path")
    
    # Test loading from non-existent path
    with pytest.raises(FileNotFoundError):
        MovieRecommender.load_model("non_existent_path") 