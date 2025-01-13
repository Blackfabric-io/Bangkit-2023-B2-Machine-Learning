#!/usr/bin/env python3
"""Main script for MNIST CNN classification."""

import argparse
import logging
from src import CNNModel, load_mnist_data, preprocess_images

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Run MNIST CNN classification model."""
    parser = argparse.ArgumentParser(description='Train and evaluate MNIST CNN classification model')
    parser.add_argument('--data-path', type=str, help='Path to MNIST data file')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--threshold', type=float, default=0.995, help='Accuracy threshold for early stopping')
    parser.add_argument('--verbose', type=int, default=1, help='Verbosity mode')
    args = parser.parse_args()
    
    try:
        # Load and preprocess data
        logger.info("Loading MNIST data...")
        x_train, y_train = load_mnist_data(args.data_path)
        
        logger.info("Preprocessing images...")
        x_train = preprocess_images(x_train)
        
        # Create and train model
        logger.info("Creating CNN model...")
        model = CNNModel()
        
        logger.info(f"Training model for {args.epochs} epochs (or until {args.threshold*100}% accuracy)...")
        history = model.train(
            x_train, 
            y_train, 
            epochs=args.epochs,
            callback_threshold=args.threshold,
            verbose=args.verbose
        )
        
        # Print final metrics
        final_epoch = len(history.history['accuracy'])
        final_accuracy = history.history['accuracy'][-1]
        final_loss = history.history['loss'][-1]
        
        logger.info(f"\nTraining completed after {final_epoch} epochs")
        logger.info(f"Final accuracy: {final_accuracy:.4f}")
        logger.info(f"Final loss: {final_loss:.4f}")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main() 