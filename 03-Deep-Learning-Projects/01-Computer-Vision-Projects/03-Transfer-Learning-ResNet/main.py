#!/usr/bin/env python3
"""Command line interface for ResNet transfer learning."""

import os
import argparse
import logging
from typing import List
import tensorflow as tf

from src.processors import train_model, evaluate_model, predict_image
from src.utils import plot_training_history

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='ResNet transfer learning for image classification'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument(
        '--data-dir',
        type=str,
        required=True,
        help='Directory containing training data'
    )
    train_parser.add_argument(
        '--num-classes',
        type=int,
        required=True,
        help='Number of target classes'
    )
    train_parser.add_argument(
        '--model-dir',
        type=str,
        default='models',
        help='Directory to save model weights'
    )
    train_parser.add_argument(
        '--image-size',
        type=int,
        nargs=2,
        default=[224, 224],
        help='Target image size (height width)'
    )
    train_parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Training batch size'
    )
    train_parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        help='Number of training epochs'
    )
    train_parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.001,
        help='Initial learning rate'
    )
    train_parser.add_argument(
        '--fine-tune',
        action='store_true',
        help='Perform fine-tuning after initial training'
    )
    train_parser.add_argument(
        '--fine-tune-epochs',
        type=int,
        default=5,
        help='Number of fine-tuning epochs'
    )
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate the model')
    eval_parser.add_argument(
        '--data-dir',
        type=str,
        required=True,
        help='Directory containing test data'
    )
    eval_parser.add_argument(
        '--weights-path',
        type=str,
        required=True,
        help='Path to model weights'
    )
    eval_parser.add_argument(
        '--num-classes',
        type=int,
        required=True,
        help='Number of classes'
    )
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Make predictions')
    predict_parser.add_argument(
        '--image-path',
        type=str,
        required=True,
        help='Path to input image'
    )
    predict_parser.add_argument(
        '--weights-path',
        type=str,
        required=True,
        help='Path to model weights'
    )
    predict_parser.add_argument(
        '--class-names',
        type=str,
        nargs='+',
        required=True,
        help='List of class names'
    )
    
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()
    
    if args.command == 'train':
        # Train model
        logger.info("Starting model training...")
        history = train_model(
            data_dir=args.data_dir,
            num_classes=args.num_classes,
            model_dir=args.model_dir,
            image_size=tuple(args.image_size),
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            fine_tune=args.fine_tune,
            fine_tune_epochs=args.fine_tune_epochs
        )
        
        # Plot training history
        plot_training_history(history)
        
    elif args.command == 'evaluate':
        # Load model
        logger.info("Loading model...")
        model = ResNetTransfer(num_classes=args.num_classes)
        model.load_weights(args.weights_path)
        
        # Evaluate
        loss, accuracy = evaluate_model(
            model=model,
            data_dir=args.data_dir
        )
        
    elif args.command == 'predict':
        # Load model
        logger.info("Loading model...")
        model = ResNetTransfer(num_classes=len(args.class_names))
        model.load_weights(args.weights_path)
        
        # Make prediction
        predicted_class, confidence = predict_image(
            model=model,
            image_path=args.image_path,
            class_names=args.class_names
        )
        
        logger.info(f"Predicted class: {predicted_class}")
        logger.info(f"Confidence: {confidence:.4f}")
        
    else:
        logger.error("No command specified")
        return 1
    
    return 0

if __name__ == '__main__':
    # Enable memory growth for GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    
    exit(main()) 