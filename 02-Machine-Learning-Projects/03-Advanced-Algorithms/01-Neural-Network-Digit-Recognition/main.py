"""
Main script for training and evaluating digit recognition model.
"""

import argparse
from pathlib import Path
import logging
from src.processors.main import DigitRecognizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train and evaluate digit recognition model.'
    )
    
    # Model parameters
    parser.add_argument('--input_size', type=int, default=784,
                       help='Size of input features')
    parser.add_argument('--hidden_sizes', type=int, nargs='+',
                       default=[128, 64],
                       help='List of hidden layer sizes')
    parser.add_argument('--learning_rate', type=float, default=0.1,
                       help='Learning rate for training')
    parser.add_argument('--activation', type=str, default='relu',
                       choices=['relu', 'tanh'],
                       help='Activation function for hidden layers')
    parser.add_argument('--weight_init', type=str, default='he',
                       choices=['he', 'xavier'],
                       help='Weight initialization method')
    
    # Training parameters
    parser.add_argument('--n_iterations', type=int, default=1000,
                       help='Number of training iterations')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Size of mini-batches')
    parser.add_argument('--eval_interval', type=int, default=100,
                       help='Interval for evaluation and plotting')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default='data',
                       help='Directory for dataset')
    
    # Mode selection
    parser.add_argument('--train', action='store_true',
                       help='Train a new model')
    parser.add_argument('--evaluate', action='store_true',
                       help='Evaluate trained model')
    
    # Model saving/loading
    parser.add_argument('--save_dir', type=str,
                       help='Directory to save trained model')
    parser.add_argument('--load_dir', type=str,
                       help='Directory to load trained model from')
    
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()
    
    # Initialize or load model
    if args.load_dir:
        logger.info("Loading model from %s", args.load_dir)
        model = DigitRecognizer.load(args.load_dir)
    else:
        model = DigitRecognizer(
            input_size=args.input_size,
            hidden_sizes=args.hidden_sizes,
            learning_rate=args.learning_rate,
            activation=args.activation,
            weight_init=args.weight_init
        )
    
    # Load data
    logger.info("Loading MNIST dataset from %s", args.data_dir)
    model.load_data(args.data_dir)
    
    # Train model if requested
    if args.train:
        logger.info("Starting training for %d iterations", args.n_iterations)
        results = model.train(
            n_iterations=args.n_iterations,
            batch_size=args.batch_size,
            eval_interval=args.eval_interval
        )
        
        # Save model if requested
        if args.save_dir:
            logger.info("Saving model to %s", args.save_dir)
            model.save(args.save_dir)
            
        logger.info("Final accuracy: %.3f", results['final_accuracy'])
    
    # Evaluate model if requested
    if args.evaluate:
        logger.info("Evaluating model")
        metrics = model.evaluate()
        
        print("\nEvaluation Results:")
        print(f"Accuracy: {metrics['accuracy']:.3f}")
        
        # Print per-class metrics
        for i in range(10):
            print(f"\nDigit {i}:")
            print(f"Precision: {metrics[f'precision_{i}']:.3f}")
            print(f"Recall: {metrics[f'recall_{i}']:.3f}")
            print(f"F1 Score: {metrics[f'f1_{i}']:.3f}")

if __name__ == '__main__':
    main() 