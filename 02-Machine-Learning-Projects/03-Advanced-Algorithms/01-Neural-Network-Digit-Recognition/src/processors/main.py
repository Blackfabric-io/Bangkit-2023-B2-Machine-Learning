"""
Main processing logic for digit recognition.
"""

from typing import Dict, List, Optional, Any
import numpy as np
from pathlib import Path
import logging
from src.core.base import NeuralNetwork, NeuralNetConfig
from src.utils.helpers import (
    download_mnist,
    load_mnist,
    preprocess_data,
    plot_digits,
    plot_training_history,
    plot_confusion_matrix,
    compute_metrics,
    save_results
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DigitRecognizer:
    """Digit recognition model trainer and evaluator."""
    
    def __init__(self, input_size: int = 784,
                 hidden_sizes: List[int] = [128, 64],
                 output_size: int = 10,
                 learning_rate: float = 0.1,
                 activation: str = 'relu',
                 weight_init: str = 'he'):
        """Initialize digit recognizer.
        
        Args:
            input_size: Size of input features.
            hidden_sizes: List of hidden layer sizes.
            output_size: Number of output classes.
            learning_rate: Learning rate for training.
            activation: Activation function for hidden layers.
            weight_init: Weight initialization method.
        """
        # Initialize network
        config = NeuralNetConfig(
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            output_size=output_size,
            learning_rate=learning_rate,
            activation=activation,
            weight_init=weight_init
        )
        self.network = NeuralNetwork(config)
        
        # Store parameters
        self.parameters = {
            'input_size': input_size,
            'hidden_sizes': hidden_sizes,
            'output_size': output_size,
            'learning_rate': learning_rate,
            'activation': activation,
            'weight_init': weight_init
        }
        
        logger.info("Initialized digit recognizer with parameters: %s",
                   self.parameters)
    
    def load_data(self, data_dir: str = 'data') -> None:
        """Load and preprocess MNIST dataset.
        
        Args:
            data_dir: Directory containing dataset.
        """
        # Download dataset if needed
        download_mnist(data_dir)
        
        # Load dataset
        X_train, y_train, X_test, y_test = load_mnist(data_dir)
        
        # Preprocess data
        self.X_train, self.y_train = preprocess_data(X_train, y_train)
        self.X_test, self.y_test = preprocess_data(X_test, y_test)
        
        # Plot sample digits
        plot_digits(X_train, y_train)
    
    def train(self, n_iterations: int = 1000,
              batch_size: int = 64,
              eval_interval: int = 100) -> Dict[str, Any]:
        """Train the neural network.
        
        Args:
            n_iterations: Number of training iterations.
            batch_size: Size of mini-batches.
            eval_interval: Interval for evaluation and plotting.
            
        Returns:
            Dictionary containing training results.
            
        Raises:
            ValueError: If data hasn't been loaded.
        """
        if not hasattr(self, 'X_train'):
            raise ValueError("Data not loaded. Call load_data() first.")
        
        costs = []
        m = self.X_train.shape[1]
        
        for i in range(n_iterations):
            # Select random mini-batch
            batch_indices = np.random.choice(m, batch_size)
            X_batch = self.X_train[:, batch_indices]
            y_batch = self.y_train[:, batch_indices]
            
            # Forward propagation
            self.network.forward_propagation(X_batch)
            
            # Compute cost
            cost = self.network.compute_cost(y_batch)
            costs.append(cost)
            
            # Backward propagation
            self.network.backward_propagation(y_batch)
            
            # Update parameters
            self.network.update_parameters()
            
            # Log progress
            if (i + 1) % eval_interval == 0:
                logger.info(
                    "Iteration %d/%d: cost=%.3f",
                    i + 1, n_iterations, cost
                )
                
                # Evaluate on validation set
                accuracy = self.evaluate(plot_results=False)['accuracy']
                logger.info("Validation accuracy: %.3f", accuracy)
                
                # Plot training progress
                plot_training_history(costs)
        
        # Final evaluation
        results = self.evaluate()
        
        # Compile results
        training_results = {
            'costs': costs,
            'parameters': self.parameters,
            'final_accuracy': results['accuracy'],
            'metrics': results
        }
        
        return training_results
    
    def evaluate(self, plot_results: bool = True) -> Dict[str, float]:
        """Evaluate model on test set.
        
        Args:
            plot_results: Whether to plot confusion matrix.
            
        Returns:
            Dictionary containing evaluation metrics.
            
        Raises:
            ValueError: If data hasn't been loaded.
        """
        if not hasattr(self, 'X_test'):
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Get predictions
        y_pred = self.predict(self.X_test)
        y_true = np.argmax(self.y_test, axis=0)
        
        # Compute metrics
        metrics = compute_metrics(y_true, y_pred)
        
        if plot_results:
            # Plot confusion matrix
            plot_confusion_matrix(y_true, y_pred)
            
            # Plot sample predictions
            plot_digits(
                self.X_test,
                y_true,
                predictions=y_pred,
                n_samples=10
            )
        
        return metrics
    
    def predict(self, X: ArrayType) -> ArrayType:
        """Make predictions for input data.
        
        Args:
            X: Input images.
            
        Returns:
            Predicted digit labels.
        """
        return self.network.predict(X)
    
    def save(self, save_dir: str) -> None:
        """Save trained model and results.
        
        Args:
            save_dir: Directory to save results.
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save network parameters
        self.network.save_parameters(save_dir / 'model_parameters.npy')
        
        # Save training parameters
        save_results(
            {'parameters': self.parameters},
            save_dir / 'training_parameters.pkl'
        )
        
        logger.info("Saved model to %s", save_dir)
    
    @classmethod
    def load(cls, load_dir: str) -> 'DigitRecognizer':
        """Load trained model.
        
        Args:
            load_dir: Directory containing saved files.
            
        Returns:
            Loaded DigitRecognizer instance.
        """
        load_dir = Path(load_dir)
        
        # Load training parameters
        with open(load_dir / 'training_parameters.pkl', 'rb') as f:
            data = pickle.load(f)
            parameters = data['parameters']
        
        # Create model with same parameters
        model = cls(
            input_size=parameters['input_size'],
            hidden_sizes=parameters['hidden_sizes'],
            output_size=parameters['output_size'],
            learning_rate=parameters['learning_rate'],
            activation=parameters['activation'],
            weight_init=parameters['weight_init']
        )
        
        # Load network parameters
        model.network.load_parameters(load_dir / 'model_parameters.npy')
        
        logger.info("Loaded model from %s", load_dir)
        return model 