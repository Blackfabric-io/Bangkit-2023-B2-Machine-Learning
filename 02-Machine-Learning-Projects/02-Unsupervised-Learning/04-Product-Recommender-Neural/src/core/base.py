"""
Core implementation of neural network recommender system.
"""

import numpy as np
import tensorflow as tf
from typing import Dict, List, Optional, Tuple, Any
import numpy.typing as npt
from dataclasses import dataclass
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Type aliases
ArrayType = npt.NDArray[np.float64]
TensorType = tf.Tensor

@dataclass
class RecommenderConfig:
    """Configuration for neural network recommender."""
    num_products: int
    num_users: int
    embedding_dim: int = 50
    hidden_units: List[int] = (256, 128)
    dropout_rate: float = 0.1
    learning_rate: float = 0.001
    batch_size: int = 64
    num_epochs: int = 10

class NeuralRecommender(tf.keras.Model):
    """Neural network-based recommender system."""
    
    def __init__(self, config: RecommenderConfig):
        """Initialize the neural recommender.
        
        Args:
            config: Model configuration parameters.
        """
        super().__init__()
        self.config = config
        
        # Create embeddings
        self.user_embedding = tf.keras.layers.Embedding(
            config.num_users,
            config.embedding_dim,
            embeddings_initializer='he_normal',
            name='user_embedding'
        )
        self.product_embedding = tf.keras.layers.Embedding(
            config.num_products,
            config.embedding_dim,
            embeddings_initializer='he_normal',
            name='product_embedding'
        )
        
        # Create dense layers
        self.dense_layers = []
        for units in config.hidden_units:
            self.dense_layers.extend([
                tf.keras.layers.Dense(
                    units,
                    activation='relu',
                    kernel_initializer='he_normal'
                ),
                tf.keras.layers.Dropout(config.dropout_rate)
            ])
        
        # Output layer
        self.output_layer = tf.keras.layers.Dense(
            1,
            activation='sigmoid',
            kernel_initializer='he_normal'
        )
        
        # Compile model
        self.compile(
            optimizer=tf.keras.optimizers.Adam(config.learning_rate),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[
                tf.keras.metrics.BinaryAccuracy(),
                tf.keras.metrics.AUC(),
                tf.keras.metrics.Precision(),
                tf.keras.metrics.Recall()
            ]
        )
        
        logger.info("Neural recommender initialized with config: %s", config)
    
    def call(self, inputs: Tuple[TensorType, TensorType],
             training: bool = False) -> TensorType:
        """Forward pass of the model.
        
        Args:
            inputs: Tuple of (user_ids, product_ids).
            training: Whether in training mode.
            
        Returns:
            Predicted interaction probabilities.
        """
        user_ids, product_ids = inputs
        
        # Get embeddings
        user_embed = self.user_embedding(user_ids)
        product_embed = self.product_embedding(product_ids)
        
        # Concatenate embeddings
        x = tf.concat([user_embed, product_embed], axis=1)
        
        # Pass through dense layers
        for layer in self.dense_layers:
            x = layer(x, training=training)
        
        # Get predictions
        return self.output_layer(x)
    
    def train_step(self, data: Tuple[Tuple[TensorType, TensorType], TensorType]
                  ) -> Dict[str, float]:
        """Custom training step.
        
        Args:
            data: Tuple of ((user_ids, product_ids), labels).
            
        Returns:
            Dictionary of metric results.
        """
        (user_ids, product_ids), labels = data
        
        with tf.GradientTape() as tape:
            # Forward pass
            y_pred = self((user_ids, product_ids), training=True)
            
            # Compute loss
            loss = self.compiled_loss(labels, y_pred)
        
        # Compute gradients and update weights
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        # Update metrics
        self.compiled_metrics.update_state(labels, y_pred)
        
        # Return metrics
        return {m.name: m.result() for m in self.metrics}
    
    def test_step(self, data: Tuple[Tuple[TensorType, TensorType], TensorType]
                 ) -> Dict[str, float]:
        """Custom test step.
        
        Args:
            data: Tuple of ((user_ids, product_ids), labels).
            
        Returns:
            Dictionary of metric results.
        """
        (user_ids, product_ids), labels = data
        
        # Forward pass
        y_pred = self((user_ids, product_ids), training=False)
        
        # Compute loss
        self.compiled_loss(labels, y_pred)
        
        # Update metrics
        self.compiled_metrics.update_state(labels, y_pred)
        
        # Return metrics
        return {m.name: m.result() for m in self.metrics}
    
    def predict_interaction(self, user_id: int, product_id: int) -> float:
        """Predict interaction probability for a user-product pair.
        
        Args:
            user_id: User ID.
            product_id: Product ID.
            
        Returns:
            Predicted interaction probability.
            
        Raises:
            ValueError: If IDs are out of range.
        """
        if not 0 <= user_id < self.config.num_users:
            raise ValueError(f"Invalid user_id: {user_id}")
        if not 0 <= product_id < self.config.num_products:
            raise ValueError(f"Invalid product_id: {product_id}")
        
        # Convert to tensors
        user_tensor = tf.convert_to_tensor([[user_id]], dtype=tf.int32)
        product_tensor = tf.convert_to_tensor([[product_id]], dtype=tf.int32)
        
        # Get prediction
        return float(self.predict((user_tensor, product_tensor))[0, 0])
    
    def recommend_products(self, user_id: int,
                         n_recommendations: int = 10,
                         exclude_interacted: bool = True,
                         interaction_matrix: Optional[ArrayType] = None
                         ) -> Tuple[ArrayType, ArrayType]:
        """Generate product recommendations for a user.
        
        Args:
            user_id: User ID.
            n_recommendations: Number of recommendations to generate.
            exclude_interacted: Whether to exclude products the user has interacted with.
            interaction_matrix: Binary interaction matrix if exclude_interacted is True.
            
        Returns:
            Tuple of (product indices, scores).
            
        Raises:
            ValueError: If user_id is invalid or interaction_matrix is missing.
        """
        if not 0 <= user_id < self.config.num_users:
            raise ValueError(f"Invalid user_id: {user_id}")
        if exclude_interacted and interaction_matrix is None:
            raise ValueError("interaction_matrix required when exclude_interacted is True")
        
        # Create user tensor
        user_tensor = tf.convert_to_tensor(
            [[user_id]] * self.config.num_products,
            dtype=tf.int32
        )
        
        # Create product tensor
        product_tensor = tf.convert_to_tensor(
            [[i] for i in range(self.config.num_products)],
            dtype=tf.int32
        )
        
        # Get predictions for all products
        scores = self.predict((user_tensor, product_tensor))[:, 0]
        
        # Exclude interacted products if requested
        if exclude_interacted:
            scores = scores.numpy()
            scores[interaction_matrix[user_id] > 0] = -np.inf
        
        # Get top recommendations
        top_indices = np.argsort(scores)[::-1][:n_recommendations]
        return top_indices, scores[top_indices] 