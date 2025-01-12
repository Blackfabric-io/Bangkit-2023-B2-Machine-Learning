"""Training module for neural network model.

This module handles:
- Model training loop
- Evaluation metrics
- Progress tracking
- Model checkpointing
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
import time
from pathlib import Path
import matplotlib.pyplot as plt

from ..core.base import NeuralNetwork
from ..data.loader import DataLoader

logger = logging.getLogger(__name__)

class ModelTrainer:
    """Trainer class for neural network model."""
    
    def __init__(self, model: NeuralNetwork, data_loader: DataLoader):
        """Initialize trainer.
        
        Args:
            model: Neural network model
            data_loader: Data loader instance
        """
        self.model = model
        self.data_loader = data_loader
        self.history: Dict[str, List[float]] = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
    def train(self, 
              num_epochs: int,
              batch_size: int,
              validation_split: float = 0.1,
              checkpoint_dir: Optional[str] = None,
              verbose: bool = True) -> Dict[str, List[float]]:
        """Train the neural network model.
        
        Args:
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            validation_split: Fraction of training data to use for validation
            checkpoint_dir: Directory to save model checkpoints
            verbose: Whether to print training progress
            
        Returns:
            Training history
        """
        logger.info("Starting model training...")
        start_time = time.time()
        
        # Load and split data
        X_train, y_train = self.data_loader.get_train_data()
        val_size = int(X_train.shape[1] * validation_split)
        train_size = X_train.shape[1] - val_size
        
        X_val = X_train[:, train_size:]
        y_val = y_train[:, train_size:]
        X_train = X_train[:, :train_size]
        y_train = y_train[:, :train_size]
        
        # Create checkpoint directory
        if checkpoint_dir:
            checkpoint_path = Path(checkpoint_dir)
            checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        best_val_loss = float('inf')
        batch_generator = self.data_loader.get_batch(batch_size)
        
        try:
            for epoch in range(num_epochs):
                epoch_start = time.time()
                train_losses = []
                train_accuracies = []
                
                # Training loop
                num_batches = train_size // batch_size
                for _ in range(num_batches):
                    X_batch, y_batch = next(batch_generator)
                    
                    # Forward pass
                    activations, cache = self.model.forward_propagation(X_batch)
                    output = activations[f'A{len(self.model.config.hidden_sizes) + 1}']
                    
                    # Compute metrics
                    batch_loss = self.model.compute_cost(output, y_batch)
                    batch_accuracy = self.compute_accuracy(output, y_batch)
                    
                    # Backward pass and update
                    gradients = self.model.backward_propagation(y_batch, activations, cache)
                    self.model.update_parameters(gradients)
                    
                    train_losses.append(batch_loss)
                    train_accuracies.append(batch_accuracy)
                
                # Validation step
                val_activations, _ = self.model.forward_propagation(X_val)
                val_output = val_activations[f'A{len(self.model.config.hidden_sizes) + 1}']
                val_loss = self.model.compute_cost(val_output, y_val)
                val_accuracy = self.compute_accuracy(val_output, y_val)
                
                # Update history
                self.history['train_loss'].append(np.mean(train_losses))
                self.history['train_accuracy'].append(np.mean(train_accuracies))
                self.history['val_loss'].append(val_loss)
                self.history['val_accuracy'].append(val_accuracy)
                
                # Save checkpoint if best model
                if checkpoint_dir and val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.model.save_model(str(checkpoint_path / 'best_model.npy'))
                
                if verbose:
                    epoch_time = time.time() - epoch_start
                    logger.info(
                        f"Epoch {epoch+1}/{num_epochs} - {epoch_time:.2f}s - "
                        f"loss: {np.mean(train_losses):.4f} - "
                        f"accuracy: {np.mean(train_accuracies):.4f} - "
                        f"val_loss: {val_loss:.4f} - "
                        f"val_accuracy: {val_accuracy:.4f}"
                    )
        
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise
        finally:
            total_time = time.time() - start_time
            logger.info(f"Training completed in {total_time:.2f}s")
            
        return self.history
    
    @staticmethod
    def compute_accuracy(predictions: np.ndarray, labels: np.ndarray) -> float:
        """Compute accuracy for predictions.
        
        Args:
            predictions: Model predictions
            labels: True labels
            
        Returns:
            Accuracy score
        """
        pred_classes = np.argmax(predictions, axis=0)
        true_classes = np.argmax(labels, axis=0)
        return float(np.mean(pred_classes == true_classes))
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """Evaluate model on test data.
        
        Args:
            X: Test features
            y: Test labels
            
        Returns:
            Tuple of (loss, accuracy)
        """
        activations, _ = self.model.forward_propagation(X)
        output = activations[f'A{len(self.model.config.hidden_sizes) + 1}']
        loss = self.model.compute_cost(output, y)
        accuracy = self.compute_accuracy(output, y)
        return loss, accuracy
    
    def plot_history(self, save_path: Optional[str] = None) -> None:
        """Plot training history.
        
        Args:
            save_path: Path to save the plot
        """
        plt.figure(figsize=(12, 4))
        
        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(self.history['train_loss'], label='Training Loss')
        plt.plot(self.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot accuracy
        plt.subplot(1, 2, 2)
        plt.plot(self.history['train_accuracy'], label='Training Accuracy')
        plt.plot(self.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.show() 