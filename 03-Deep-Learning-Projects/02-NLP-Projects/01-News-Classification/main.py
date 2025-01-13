#!/usr/bin/env python3
"""Command line interface for BBC News Classification."""

import os
import argparse
import logging
from typing import List
import tensorflow as tf

from src.processors import process_data, train_model, evaluate_model
from src.utils import plot_metrics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='BBC News Classification using NLP'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument(
        '--data-file',
        type=str,
        required=True,
        help='Path to training data CSV file'
    )
    train_parser.add_argument(
        '--model-dir',
        type=str,
        default='models',
        help='Directory to save model'
    )
    train_parser.add_argument(
        '--embedding-dim',
        type=int,
        default=100,
        help='Embedding dimension'
    )
    train_parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        help='Number of training epochs'
    )
    train_parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Training batch size'
    )
    train_parser.add_argument(
        '--validation-split',
        type=float,
        default=0.2,
        help='Fraction of data to use for validation'
    )
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate the model')
    eval_parser.add_argument(
        '--data-file',
        type=str,
        required=True,
        help='Path to test data CSV file'
    )
    eval_parser.add_argument(
        '--model-path',
        type=str,
        required=True,
        help='Path to saved model'
    )
    eval_parser.add_argument(
        '--class-names',
        type=str,
        nargs='+',
        default=['tech', 'business', 'sport', 'entertainment', 'politics'],
        help='List of class names'
    )
    
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()
    
    if args.command == 'train':
        # Process data
        logger.info("Processing training data...")
        data, tokenizer = process_data(args.data_file)
        
        # Create model directory
        os.makedirs(args.model_dir, exist_ok=True)
        
        # Train model
        logger.info("Training model...")
        model, history = train_model(
            train_sequences=data['train_sequences'],
            train_labels=data['train_labels'],
            num_classes=len(tokenizer.label_word_index),
            embedding_dim=args.embedding_dim,
            epochs=args.epochs,
            batch_size=args.batch_size,
            validation_split=args.validation_split
        )
        
        # Save model
        model_path = os.path.join(args.model_dir, 'news_classifier.h5')
        model.save(model_path)
        logger.info(f"Model saved to {model_path}")
        
    elif args.command == 'evaluate':
        # Load model
        logger.info("Loading model...")
        model = tf.keras.models.load_model(args.model_path)
        
        # Process test data
        logger.info("Processing test data...")
        data, tokenizer = process_data(args.data_file)
        
        # Evaluate model
        loss, accuracy = evaluate_model(
            model=model,
            test_sequences=data['train_sequences'],
            test_labels=data['train_labels'],
            class_names=args.class_names
        )
        
    else:
        logger.error("No command specified")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main()) 