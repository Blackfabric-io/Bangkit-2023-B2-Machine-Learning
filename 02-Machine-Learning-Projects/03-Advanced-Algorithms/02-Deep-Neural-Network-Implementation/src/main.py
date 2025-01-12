"""Main script demonstrating neural network usage for digit recognition.

This script shows how to:
1. Load and preprocess MNIST data
2. Create and configure the neural network
3. Train the model
4. Evaluate performance
5. Make predictions
"""

import logging
from pathlib import Path
import argparse

from core.base import NeuralNetwork, NeuralNetConfig
from data.loader import DataLoader
from training.trainer import ModelTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train neural network for digit recognition')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden-sizes', type=int, nargs='+', default=[25, 15],
                       help='Hidden layer sizes')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                       help='Directory to save model checkpoints')
    return parser.parse_args()

def main():
    """Main function to run the neural network training pipeline."""
    args = parse_args()
    
    try:
        # Create checkpoint directory
        checkpoint_dir = Path(args.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Load and preprocess data
        logger.info("Loading data...")
        data_loader = DataLoader(flatten=True, normalize=True)
        data_loader.load_mnist()
        
        # Create and configure model
        logger.info("Creating model...")
        config = NeuralNetConfig(
            input_size=400,  # 20x20 pixels
            hidden_sizes=args.hidden_sizes,
            output_size=10,  # 10 digits
            learning_rate=args.learning_rate,
            num_iterations=args.epochs * (60000 // args.batch_size),  # Total number of weight updates
            batch_size=args.batch_size
        )
        model = NeuralNetwork(config)
        
        # Create trainer
        trainer = ModelTrainer(model, data_loader)
        
        # Train model
        logger.info("Starting training...")
        history = trainer.train(
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            checkpoint_dir=args.checkpoint_dir,
            verbose=True
        )
        
        # Plot training history
        trainer.plot_history(save_path=str(checkpoint_dir / 'training_history.png'))
        
        # Evaluate on test set
        logger.info("Evaluating model...")
        X_test, y_test = data_loader.get_test_data()
        test_loss, test_accuracy = trainer.evaluate(X_test, y_test)
        logger.info(f"Test Loss: {test_loss:.4f}")
        logger.info(f"Test Accuracy: {test_accuracy:.4f}")
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == '__main__':
    main() 