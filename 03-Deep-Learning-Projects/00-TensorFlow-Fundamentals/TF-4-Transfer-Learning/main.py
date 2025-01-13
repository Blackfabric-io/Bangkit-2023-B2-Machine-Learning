#!/usr/bin/env python3
"""Command line interface for training and evaluating the emotion classification model."""

import os
import argparse
import logging
from datetime import datetime

from src.core.base import EmotionModel, AccuracyThresholdCallback
from src.utils.helpers import create_data_generator, validate_image_directory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train and evaluate emotion classification model'
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        required=True,
        help='Directory containing happy/ and sad/ subdirectories'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=15,
        help='Maximum number of training epochs'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=10,
        help='Batch size for training'
    )
    
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.999,
        help='Accuracy threshold to stop training'
    )
    
    parser.add_argument(
        '--model-dir',
        type=str,
        default='models',
        help='Directory to save model weights'
    )
    
    parser.add_argument(
        '--verbose',
        type=int,
        choices=[0, 1, 2],
        default=1,
        help='Verbosity level for training'
    )
    
    return parser.parse_args()

def main():
    """Main training function."""
    args = parse_args()
    
    # Validate data directory
    logger.info("Validating data directory...")
    try:
        stats = validate_image_directory(args.data_dir)
        logger.info("Found %d happy and %d sad images", 
                   stats['happy'], stats['sad'])
    except ValueError as e:
        logger.error("Data directory validation failed: %s", str(e))
        return 1
        
    # Create data generator
    logger.info("Creating data generator...")
    try:
        train_generator = create_data_generator(
            args.data_dir,
            batch_size=args.batch_size
        )
    except Exception as e:
        logger.error("Failed to create data generator: %s", str(e))
        return 1
    
    # Create and train model
    logger.info("Initializing model...")
    model = EmotionModel()
    callback = AccuracyThresholdCallback(threshold=args.threshold)
    
    logger.info("Starting training...")
    try:
        history = model.train(
            train_generator,
            epochs=args.epochs,
            callbacks=[callback],
            verbose=args.verbose
        )
    except Exception as e:
        logger.error("Training failed: %s", str(e))
        return 1
    
    # Save model
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
        
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(
        args.model_dir,
        f"emotion_model_{timestamp}.h5"
    )
    
    logger.info("Saving model to %s", model_path)
    try:
        model.save(model_path)
    except Exception as e:
        logger.error("Failed to save model: %s", str(e))
        return 1
    
    # Print training summary
    final_epoch = len(history.history['accuracy'])
    final_acc = history.history['accuracy'][-1]
    final_loss = history.history['loss'][-1]
    
    logger.info("Training completed:")
    logger.info("- Epochs completed: %d", final_epoch)
    logger.info("- Final accuracy: %.4f", final_acc)
    logger.info("- Final loss: %.4f", final_loss)
    
    return 0

if __name__ == '__main__':
    exit(main()) 