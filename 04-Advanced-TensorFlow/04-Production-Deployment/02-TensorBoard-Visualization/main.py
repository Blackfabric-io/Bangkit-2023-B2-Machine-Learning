"""
Main entry point for Fashion MNIST training with TensorBoard visualization.
"""

import argparse
import logging
from src.core.model import FashionMNISTModel
from src.utils.visualization import TensorBoardLogger

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train Fashion MNIST model with TensorBoard visualization'
    )
    
    parser.add_argument(
        '--log-dir',
        type=str,
        default='logs/fashion_mnist',
        help='Directory to store TensorBoard logs'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=5,
        help='Number of training epochs'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Training batch size'
    )
    
    parser.add_argument(
        '--validation-split',
        type=float,
        default=0.2,
        help='Fraction of data to use for validation'
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
        logger.info("Initializing Fashion MNIST model...")
        model = FashionMNISTModel()
        
        # Initialize TensorBoard logger
        logger.info("Setting up TensorBoard logging...")
        tb_logger = TensorBoardLogger(
            log_dir=args.log_dir,
            class_names=FashionMNISTModel.CLASS_NAMES
        )
        
        # Train model
        logger.info("Starting model training...")
        model.train(
            epochs=args.epochs,
            batch_size=args.batch_size,
            validation_split=args.validation_split,
            callbacks=[
                tb_logger.tensorboard_callback,
                tb_logger.confusion_matrix_callback
            ]
        )
        
        # Evaluate model
        logger.info("Evaluating model...")
        model.evaluate(batch_size=args.batch_size)
        
        logger.info(f"Training completed. View results with: tensorboard --logdir {args.log_dir}")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main() 