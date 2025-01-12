"""
Main script for training and using the neural network recommender system.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from src.processors.main import ProductRecommender

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train and use a neural network recommender system.'
    )
    
    # Data paths
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to interaction data file')
    
    # Model parameters
    parser.add_argument('--embedding_dim', type=int, default=50,
                       help='Dimension of embedding vectors')
    parser.add_argument('--hidden_units', type=int, nargs='+', default=[256, 128],
                       help='Hidden layer sizes')
    parser.add_argument('--dropout_rate', type=float, default=0.1,
                       help='Dropout rate for regularization')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate for optimization')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=10,
                       help='Number of training epochs')
    
    # Training options
    parser.add_argument('--train', action='store_true',
                       help='Train a new model')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Fraction of data to use for testing')
    parser.add_argument('--random_state', type=int, default=None,
                       help='Random seed for reproducibility')
    parser.add_argument('--save_model', type=str,
                       help='Directory to save trained model')
    parser.add_argument('--load_model', type=str,
                       help='Directory to load trained model from')
    
    # Recommendation options
    parser.add_argument('--recommend', action='store_true',
                       help='Generate recommendations for users')
    parser.add_argument('--user_ids', type=str, nargs='+',
                       help='User IDs to generate recommendations for')
    parser.add_argument('--n_recommendations', type=int, default=10,
                       help='Number of recommendations to generate')
    
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()
    
    # Initialize or load recommender
    if args.load_model:
        print(f"\nLoading model from {args.load_model}...")
        recommender = ProductRecommender.load(args.load_model)
    else:
        recommender = ProductRecommender(
            embedding_dim=args.embedding_dim,
            hidden_units=args.hidden_units,
            dropout_rate=args.dropout_rate,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs
        )
    
    # Load data and train model if requested
    if args.train:
        # Load and analyze data
        recommender.load_data(args.data_path)
        
        # Train model
        history = recommender.train(
            test_size=args.test_size,
            random_state=args.random_state
        )
        
        # Save model if requested
        if args.save_model:
            print(f"\nSaving model to {args.save_model}...")
            recommender.save(args.save_model)
    
    # Generate recommendations if requested
    if args.recommend:
        if args.user_ids is None:
            print("\nNo user IDs provided for recommendations.")
            return
        
        print("\nGenerating recommendations...")
        for user_id in args.user_ids:
            try:
                recommendations = recommender.recommend_for_user(
                    user_id,
                    n_recommendations=args.n_recommendations
                )
                
                print(f"\nTop {args.n_recommendations} recommendations for user {user_id}:")
                for _, row in recommendations.iterrows():
                    print(f"{row['Product']} (Score: {row['Score']:.3f})")
                    
            except ValueError as e:
                print(f"Error generating recommendations for user {user_id}: {e}")

if __name__ == '__main__':
    main() 