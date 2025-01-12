"""
Main processing logic for neural network recommender system.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow as tf
import logging
from src.core.base import NeuralRecommender, RecommenderConfig
from src.utils.helpers import (
    load_interaction_data,
    create_train_test_split,
    plot_training_history,
    plot_interaction_heatmap,
    compute_interaction_stats,
    format_recommendations,
    save_model,
    load_model
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProductRecommender:
    """Product recommender system using neural networks."""
    
    def __init__(self, embedding_dim: int = 50,
                 hidden_units: List[int] = (256, 128),
                 dropout_rate: float = 0.1,
                 learning_rate: float = 0.001,
                 batch_size: int = 64,
                 num_epochs: int = 10):
        """Initialize the product recommender.
        
        Args:
            embedding_dim: Dimension of embedding vectors.
            hidden_units: List of hidden layer sizes.
            dropout_rate: Dropout rate for regularization.
            learning_rate: Learning rate for optimization.
            batch_size: Batch size for training.
            num_epochs: Number of training epochs.
        """
        self.config = {
            'embedding_dim': embedding_dim,
            'hidden_units': hidden_units,
            'dropout_rate': dropout_rate,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'num_epochs': num_epochs
        }
        
        self.model: Optional[NeuralRecommender] = None
        self.user_to_idx: Optional[Dict[str, int]] = None
        self.product_to_idx: Optional[Dict[str, int]] = None
        self.interaction_matrix: Optional[np.ndarray] = None
    
    def load_data(self, data_path: str, analyze: bool = True) -> Dict[str, float]:
        """Load and preprocess interaction data.
        
        Args:
            data_path: Path to interaction data file.
            analyze: Whether to analyze and plot data statistics.
            
        Returns:
            Dictionary containing interaction statistics.
        """
        # Load data
        logger.info("Loading interaction data from %s", data_path)
        self.interaction_matrix, self.user_to_idx, self.product_to_idx = \
            load_interaction_data(data_path)
        
        # Compute statistics
        stats = compute_interaction_stats(self.interaction_matrix)
        
        if analyze:
            # Print statistics
            print("\nDataset Statistics:")
            print(f"Number of users: {stats['n_users']}")
            print(f"Number of products: {stats['n_products']}")
            print(f"Number of interactions: {stats['n_interactions']}")
            print(f"Matrix sparsity: {stats['sparsity']*100:.1f}%")
            print(f"Mean interactions per user: {stats['mean_user_interactions']:.1f}")
            print(f"Mean interactions per product: {stats['mean_product_interactions']:.1f}")
            
            # Plot interaction heatmap
            plot_interaction_heatmap(self.interaction_matrix)
        
        return stats
    
    def train(self, test_size: float = 0.2,
              random_state: Optional[int] = None,
              validation_split: float = 0.1) -> Dict[str, List[float]]:
        """Train the recommender system.
        
        Args:
            test_size: Fraction of data to use for testing.
            random_state: Random seed for reproducibility.
            validation_split: Fraction of training data to use for validation.
            
        Returns:
            Dictionary containing training history.
            
        Raises:
            ValueError: If data hasn't been loaded.
        """
        if self.interaction_matrix is None:
            raise ValueError("Must load data before training")
        
        # Create train/test split
        train_dataset, test_dataset = create_train_test_split(
            self.interaction_matrix,
            test_size=test_size,
            random_state=random_state
        )
        
        # Configure datasets
        train_dataset = train_dataset.shuffle(10000).batch(self.config['batch_size'])
        test_dataset = test_dataset.batch(self.config['batch_size'])
        
        # Initialize model
        model_config = RecommenderConfig(
            num_users=len(self.user_to_idx),
            num_products=len(self.product_to_idx),
            embedding_dim=self.config['embedding_dim'],
            hidden_units=self.config['hidden_units'],
            dropout_rate=self.config['dropout_rate'],
            learning_rate=self.config['learning_rate'],
            batch_size=self.config['batch_size'],
            num_epochs=self.config['num_epochs']
        )
        self.model = NeuralRecommender(model_config)
        
        # Train model
        logger.info("Training model...")
        history = self.model.fit(
            train_dataset,
            epochs=self.config['num_epochs'],
            validation_data=test_dataset,
            verbose=1
        )
        
        # Plot training history
        plot_training_history(history.history)
        
        return history.history
    
    def recommend_for_user(self, user_id: str,
                          n_recommendations: int = 10,
                          exclude_interacted: bool = True) -> pd.DataFrame:
        """Generate product recommendations for a user.
        
        Args:
            user_id: User ID.
            n_recommendations: Number of recommendations to generate.
            exclude_interacted: Whether to exclude products the user has interacted with.
            
        Returns:
            DataFrame containing recommendations.
            
        Raises:
            ValueError: If model hasn't been trained or user_id is invalid.
        """
        if self.model is None:
            raise ValueError("Must train model before making recommendations")
        if user_id not in self.user_to_idx:
            raise ValueError(f"Invalid user_id: {user_id}")
        
        # Get user index
        user_idx = self.user_to_idx[user_id]
        
        # Get recommendations
        product_indices, scores = self.model.recommend_products(
            user_idx,
            n_recommendations=n_recommendations,
            exclude_interacted=exclude_interacted,
            interaction_matrix=self.interaction_matrix
        )
        
        # Create reverse mapping
        idx_to_product = {idx: id_ for id_, idx in self.product_to_idx.items()}
        
        # Format recommendations
        return format_recommendations(product_indices, scores, idx_to_product)
    
    def save(self, save_dir: str) -> None:
        """Save trained model and mappings.
        
        Args:
            save_dir: Directory to save files.
            
        Raises:
            ValueError: If model hasn't been trained.
        """
        if self.model is None:
            raise ValueError("Must train model before saving")
            
        save_model(
            self.model,
            self.user_to_idx,
            self.product_to_idx,
            save_dir
        )
    
    @classmethod
    def load(cls, load_dir: str) -> 'ProductRecommender':
        """Load trained model and mappings.
        
        Args:
            load_dir: Directory containing saved files.
            
        Returns:
            Loaded ProductRecommender instance.
            
        Raises:
            FileNotFoundError: If files don't exist.
        """
        # Create instance
        instance = cls()
        
        # Load model and mappings
        instance.model, instance.user_to_idx, instance.product_to_idx = \
            load_model(load_dir)
        
        return instance 