"""
Main entry point for MNIST model training and export.
"""

import argparse
import logging
from src.core.model import MNIST

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train and export MNIST model in SavedModel format'
    )
    
    parser.add_argument(
        '--export-path',
        type=str,
        default='./saved_model',
        help='Path to export SavedModel'
    )
    
    parser.add_argument(
        '--buffer-size',
        type=int,
        default=1000,
        help='Shuffle buffer size for training'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for training and evaluation'
    )
    
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=1e-3,
        help='Learning rate for Adam optimizer'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=5,
        help='Number of training epochs'
    )
    
    return parser.parse_args()

def main():
    """Main execution function."""
    # Parse arguments
    args = parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize model
        logger.info("Initializing MNIST model...")
        mnist = MNIST(
            export_path=args.export_path,
            buffer_size=args.buffer_size,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            epochs=args.epochs
        )
        
        # Train model
        logger.info("Starting model training...")
        mnist.train()
        
        # Test model
        logger.info("Evaluating model...")
        mnist.test()
        
        # Export model
        logger.info("Exporting model...")
        mnist.export_model()
        
        logger.info("Process completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main() 