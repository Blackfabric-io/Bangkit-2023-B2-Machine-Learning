"""
Utility functions for data processing and visualization.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import numpy.typing as npt
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Type aliases
ArrayType = npt.NDArray[np.float64]

def load_interaction_data(filepath: str) -> Tuple[ArrayType, Dict[str, int], Dict[str, int]]:
    """Load user-product interaction data.
    
    Args:
        filepath: Path to interaction data CSV file.
        
    Returns:
        Tuple containing:
            Binary interaction matrix.
            Dictionary mapping user IDs to indices.
            Dictionary mapping product IDs to indices.
            
    Raises:
        FileNotFoundError: If data file doesn't exist.
    """
    try:
        # Load data
        df = pd.read_csv(filepath)
        
        # Create ID mappings
        user_ids = sorted(df['user_id'].unique())
        product_ids = sorted(df['product_id'].unique())
        
        user_to_idx = {id_: i for i, id_ in enumerate(user_ids)}
        product_to_idx = {id_: i for i, id_ in enumerate(product_ids)}
        
        # Create interaction matrix
        n_users = len(user_ids)
        n_products = len(product_ids)
        interaction_matrix = np.zeros((n_users, n_products))
        
        for _, row in df.iterrows():
            user_idx = user_to_idx[row['user_id']]
            product_idx = product_to_idx[row['product_id']]
            interaction_matrix[user_idx, product_idx] = 1
        
        logger.info("Loaded interaction data: %d users, %d products, %d interactions",
                   n_users, n_products, len(df))
        
        return interaction_matrix, user_to_idx, product_to_idx
    
    except FileNotFoundError:
        raise FileNotFoundError(f"Interaction data file not found at {filepath}")

def create_train_test_split(interaction_matrix: ArrayType,
                          test_size: float = 0.2,
                          random_state: Optional[int] = None
                          ) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """Create training and test datasets.
    
    Args:
        interaction_matrix: Binary interaction matrix.
        test_size: Fraction of data to use for testing.
        random_state: Random seed for reproducibility.
        
    Returns:
        Tuple of (train_dataset, test_dataset).
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Get positive and negative interactions
    pos_users, pos_products = np.where(interaction_matrix > 0)
    neg_users, neg_products = np.where(interaction_matrix == 0)
    
    # Sample negative interactions to balance dataset
    n_neg = len(pos_users)
    neg_indices = np.random.choice(len(neg_users), n_neg, replace=False)
    neg_users = neg_users[neg_indices]
    neg_products = neg_products[neg_indices]
    
    # Combine positive and negative samples
    all_users = np.concatenate([pos_users, neg_users])
    all_products = np.concatenate([pos_products, neg_products])
    all_labels = np.concatenate([np.ones(n_neg), np.zeros(n_neg)])
    
    # Shuffle data
    indices = np.random.permutation(len(all_users))
    all_users = all_users[indices]
    all_products = all_products[indices]
    all_labels = all_labels[indices]
    
    # Split into train and test
    n_test = int(len(all_users) * test_size)
    
    train_users = all_users[n_test:]
    train_products = all_products[n_test:]
    train_labels = all_labels[n_test:]
    
    test_users = all_users[:n_test]
    test_products = all_products[:n_test]
    test_labels = all_labels[:n_test]
    
    # Create TensorFlow datasets
    train_dataset = tf.data.Dataset.from_tensor_slices(
        ((train_users, train_products), train_labels)
    )
    test_dataset = tf.data.Dataset.from_tensor_slices(
        ((test_users, test_products), test_labels)
    )
    
    logger.info("Created datasets: %d training samples, %d test samples",
                len(train_users), len(test_users))
    
    return train_dataset, test_dataset

def plot_training_history(history: Dict[str, List[float]],
                         metrics: Optional[List[str]] = None) -> None:
    """Plot training history.
    
    Args:
        history: Dictionary containing training history.
        metrics: List of metrics to plot. If None, plot all metrics.
    """
    if metrics is None:
        metrics = [key for key in history.keys() if not key.startswith('val_')]
    
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 4))
    if n_metrics == 1:
        axes = [axes]
    
    for ax, metric in zip(axes, metrics):
        ax.plot(history[metric], label='Train')
        if f'val_{metric}' in history:
            ax.plot(history[f'val_{metric}'], label='Validation')
        ax.set_title(metric.capitalize())
        ax.set_xlabel('Epoch')
        ax.grid(True)
        ax.legend()
    
    plt.tight_layout()
    plt.show()

def plot_interaction_heatmap(interaction_matrix: ArrayType,
                           max_users: int = 50,
                           max_products: int = 50) -> None:
    """Plot interaction matrix heatmap.
    
    Args:
        interaction_matrix: Binary interaction matrix.
        max_users: Maximum number of users to plot.
        max_products: Maximum number of products to plot.
    """
    # Subsample matrix if needed
    plot_matrix = interaction_matrix[:max_users, :max_products]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(plot_matrix, cmap='YlOrRd', cbar_kws={'label': 'Interaction'})
    plt.title('User-Product Interactions')
    plt.xlabel('Product ID')
    plt.ylabel('User ID')
    plt.show()

def compute_interaction_stats(interaction_matrix: ArrayType) -> Dict[str, float]:
    """Compute statistics about user-product interactions.
    
    Args:
        interaction_matrix: Binary interaction matrix.
        
    Returns:
        Dictionary containing interaction statistics.
    """
    n_users, n_products = interaction_matrix.shape
    n_interactions = np.sum(interaction_matrix)
    sparsity = 1 - (n_interactions / (n_users * n_products))
    
    user_interactions = np.sum(interaction_matrix, axis=1)
    product_interactions = np.sum(interaction_matrix, axis=0)
    
    stats = {
        'n_users': n_users,
        'n_products': n_products,
        'n_interactions': int(n_interactions),
        'sparsity': float(sparsity),
        'mean_user_interactions': float(np.mean(user_interactions)),
        'mean_product_interactions': float(np.mean(product_interactions)),
        'max_user_interactions': int(np.max(user_interactions)),
        'max_product_interactions': int(np.max(product_interactions))
    }
    
    return stats

def format_recommendations(product_indices: ArrayType,
                         scores: ArrayType,
                         product_names: Dict[int, str]) -> pd.DataFrame:
    """Format product recommendations into a DataFrame.
    
    Args:
        product_indices: Indices of recommended products.
        scores: Recommendation scores.
        product_names: Dictionary mapping product indices to names.
        
    Returns:
        DataFrame containing recommendations.
    """
    recommendations = pd.DataFrame({
        'Product': [product_names.get(idx, f"Product {idx}") for idx in product_indices],
        'Score': scores
    })
    
    return recommendations.sort_values('Score', ascending=False)

def save_model(model: tf.keras.Model,
              user_to_idx: Dict[str, int],
              product_to_idx: Dict[str, int],
              save_dir: str) -> None:
    """Save model and ID mappings.
    
    Args:
        model: Trained model.
        user_to_idx: User ID to index mapping.
        product_to_idx: Product ID to index mapping.
        save_dir: Directory to save files.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model.save(save_dir / 'model')
    
    # Save mappings
    pd.to_pickle(user_to_idx, save_dir / 'user_to_idx.pkl')
    pd.to_pickle(product_to_idx, save_dir / 'product_to_idx.pkl')
    
    logger.info("Saved model and mappings to %s", save_dir)

def load_model(load_dir: str) -> Tuple[tf.keras.Model, Dict[str, int], Dict[str, int]]:
    """Load model and ID mappings.
    
    Args:
        load_dir: Directory containing saved files.
        
    Returns:
        Tuple of (model, user_to_idx, product_to_idx).
        
    Raises:
        FileNotFoundError: If files don't exist.
    """
    load_dir = Path(load_dir)
    
    try:
        # Load model
        model = tf.keras.models.load_model(load_dir / 'model')
        
        # Load mappings
        user_to_idx = pd.read_pickle(load_dir / 'user_to_idx.pkl')
        product_to_idx = pd.read_pickle(load_dir / 'product_to_idx.pkl')
        
        logger.info("Loaded model and mappings from %s", load_dir)
        
        return model, user_to_idx, product_to_idx
    
    except FileNotFoundError:
        raise FileNotFoundError(f"Model files not found in {load_dir}") 