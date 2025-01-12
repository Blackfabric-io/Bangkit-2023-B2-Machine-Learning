"""
Unit tests for neural network recommender system.
"""

import pytest
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
import tempfile
from src.core.base import NeuralRecommender, RecommenderConfig
from src.processors.main import ProductRecommender
from src.utils.helpers import (
    load_interaction_data,
    create_train_test_split,
    compute_interaction_stats,
    format_recommendations,
    save_model,
    load_model
)

def create_synthetic_data(n_users: int = 100,
                         n_products: int = 50,
                         sparsity: float = 0.95,
                         random_state: int = 42) -> pd.DataFrame:
    """Create synthetic interaction data for testing.
    
    Args:
        n_users: Number of users.
        n_products: Number of products.
        sparsity: Fraction of missing interactions.
        random_state: Random seed.
        
    Returns:
        DataFrame containing interaction data.
    """
    np.random.seed(random_state)
    
    # Generate random interactions
    n_interactions = int((1 - sparsity) * n_users * n_products)
    user_ids = np.random.randint(0, n_users, n_interactions)
    product_ids = np.random.randint(0, n_products, n_interactions)
    
    # Create DataFrame
    df = pd.DataFrame({
        'user_id': [f"user_{i}" for i in user_ids],
        'product_id': [f"product_{i}" for i in product_ids]
    }).drop_duplicates()
    
    return df

def test_neural_recommender_initialization():
    """Test initialization of NeuralRecommender class."""
    config = RecommenderConfig(num_users=100, num_products=50)
    model = NeuralRecommender(config)
    
    assert isinstance(model, tf.keras.Model)
    assert len(model.layers) > 0
    assert model.user_embedding.input_dim == 100
    assert model.product_embedding.input_dim == 50
    assert model.user_embedding.output_dim == config.embedding_dim

def test_neural_recommender_forward_pass():
    """Test forward pass of NeuralRecommender."""
    config = RecommenderConfig(num_users=100, num_products=50)
    model = NeuralRecommender(config)
    
    # Create sample batch
    user_ids = tf.constant([[0], [1], [2]])
    product_ids = tf.constant([[0], [1], [2]])
    
    # Forward pass
    outputs = model((user_ids, product_ids))
    
    assert outputs.shape == (3, 1)
    assert np.all(outputs >= 0) and np.all(outputs <= 1)

def test_product_recommender_initialization():
    """Test initialization of ProductRecommender class."""
    recommender = ProductRecommender(
        embedding_dim=32,
        hidden_units=[64, 32]
    )
    
    assert recommender.config['embedding_dim'] == 32
    assert recommender.config['hidden_units'] == [64, 32]
    assert recommender.model is None
    assert recommender.user_to_idx is None
    assert recommender.product_to_idx is None
    assert recommender.interaction_matrix is None

def test_data_loading(tmp_path):
    """Test data loading in ProductRecommender."""
    # Create synthetic data
    df = create_synthetic_data()
    data_path = tmp_path / "interactions.csv"
    df.to_csv(data_path, index=False)
    
    # Load data
    recommender = ProductRecommender()
    stats = recommender.load_data(str(data_path), analyze=False)
    
    assert isinstance(stats, dict)
    assert 'n_users' in stats
    assert 'n_products' in stats
    assert recommender.interaction_matrix is not None
    assert recommender.user_to_idx is not None
    assert recommender.product_to_idx is not None

def test_model_training():
    """Test model training in ProductRecommender."""
    # Create synthetic data
    df = create_synthetic_data(n_users=20, n_products=10)
    
    # Initialize recommender
    recommender = ProductRecommender(
        embedding_dim=16,
        hidden_units=[32],
        num_epochs=2
    )
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(suffix='.csv') as tmp:
        # Save and load data
        df.to_csv(tmp.name, index=False)
        recommender.load_data(tmp.name, analyze=False)
        
        # Train model
        history = recommender.train(random_state=42)
        
        assert isinstance(history, dict)
        assert 'loss' in history
        assert len(history['loss']) == 2
        assert history['loss'][1] <= history['loss'][0]

def test_recommendation_generation():
    """Test recommendation generation in ProductRecommender."""
    # Create synthetic data
    df = create_synthetic_data(n_users=20, n_products=10)
    
    # Initialize recommender
    recommender = ProductRecommender(
        embedding_dim=16,
        hidden_units=[32],
        num_epochs=2
    )
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(suffix='.csv') as tmp:
        # Save and load data
        df.to_csv(tmp.name, index=False)
        recommender.load_data(tmp.name, analyze=False)
        
        # Train model
        recommender.train(random_state=42)
        
        # Get recommendations
        user_id = df['user_id'].iloc[0]
        recommendations = recommender.recommend_for_user(
            user_id,
            n_recommendations=5
        )
        
        assert isinstance(recommendations, pd.DataFrame)
        assert len(recommendations) == 5
        assert 'Product' in recommendations.columns
        assert 'Score' in recommendations.columns
        assert recommendations['Score'].is_monotonic_decreasing

def test_model_save_load(tmp_path):
    """Test model saving and loading in ProductRecommender."""
    # Create synthetic data
    df = create_synthetic_data(n_users=20, n_products=10)
    
    # Initialize recommender
    recommender = ProductRecommender(
        embedding_dim=16,
        hidden_units=[32],
        num_epochs=2
    )
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(suffix='.csv') as tmp:
        # Save and load data
        df.to_csv(tmp.name, index=False)
        recommender.load_data(tmp.name, analyze=False)
        
        # Train model
        recommender.train(random_state=42)
        
        # Save model
        save_dir = tmp_path / "model"
        recommender.save(str(save_dir))
        
        # Load model
        loaded = ProductRecommender.load(str(save_dir))
        
        # Compare recommendations
        user_id = df['user_id'].iloc[0]
        original_recs = recommender.recommend_for_user(user_id)
        loaded_recs = loaded.recommend_for_user(user_id)
        
        pd.testing.assert_frame_equal(original_recs, loaded_recs)

def test_error_handling():
    """Test error handling in ProductRecommender."""
    recommender = ProductRecommender()
    
    # Test recommendation without training
    with pytest.raises(ValueError, match="Must train model"):
        recommender.recommend_for_user("user_1")
    
    # Test saving without training
    with pytest.raises(ValueError, match="Must train model"):
        recommender.save("dummy_path")
    
    # Test training without data
    with pytest.raises(ValueError, match="Must load data"):
        recommender.train()
    
    # Test loading from non-existent path
    with pytest.raises(FileNotFoundError):
        ProductRecommender.load("non_existent_path") 