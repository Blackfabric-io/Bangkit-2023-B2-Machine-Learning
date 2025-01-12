"""
Main script for training and using the movie recommender system.
"""

import argparse
from pathlib import Path
import numpy as np
from src.processors.main import MovieRecommender

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train and use a movie recommender system.'
    )
    
    # Data paths
    parser.add_argument('--ratings_path', type=str, required=True,
                       help='Path to ratings data file')
    parser.add_argument('--movie_list_path', type=str, required=True,
                       help='Path to movie list file')
    
    # Model parameters
    parser.add_argument('--num_features', type=int, default=100,
                       help='Number of latent features')
    parser.add_argument('--learning_rate', type=float, default=0.1,
                       help='Learning rate for gradient descent')
    parser.add_argument('--num_iterations', type=int, default=200,
                       help='Number of training iterations')
    parser.add_argument('--lambda_', type=float, default=1.0,
                       help='Regularization parameter')
    
    # Training options
    parser.add_argument('--train', action='store_true',
                       help='Train a new model')
    parser.add_argument('--save_model', type=str,
                       help='Directory to save trained model')
    parser.add_argument('--load_model', type=str,
                       help='Directory to load trained model from')
    
    # Recommendation options
    parser.add_argument('--recommend', action='store_true',
                       help='Generate recommendations for a new user')
    parser.add_argument('--n_recommendations', type=int, default=10,
                       help='Number of recommendations to generate')
    
    return parser.parse_args()

def get_user_ratings(movie_list):
    """Get movie ratings from user input."""
    print("\nPlease rate some movies (1-5), or press Enter to skip:")
    user_ratings = np.zeros(len(movie_list))
    
    # Show a subset of movies for rating
    sample_indices = np.random.choice(
        len(movie_list), size=min(20, len(movie_list)), replace=False
    )
    
    for idx in sample_indices:
        while True:
            rating = input(f"{movie_list[idx]}: ")
            if rating == "":
                break
            try:
                rating = float(rating)
                if 1 <= rating <= 5:
                    user_ratings[idx] = rating
                    break
                else:
                    print("Please enter a rating between 1 and 5")
            except ValueError:
                print("Please enter a valid number")
    
    return user_ratings

def main():
    """Main function."""
    args = parse_args()
    
    # Initialize or load recommender
    if args.load_model:
        print(f"\nLoading model from {args.load_model}...")
        recommender = MovieRecommender.load_model(
            args.load_model,
            hyperparameters={
                'num_features': args.num_features,
                'learning_rate': args.learning_rate,
                'num_iterations': args.num_iterations,
                'lambda_': args.lambda_
            }
        )
    else:
        recommender = MovieRecommender(
            num_features=args.num_features,
            learning_rate=args.learning_rate,
            num_iterations=args.num_iterations,
            lambda_=args.lambda_
        )
    
    # Load data and train model if requested
    if args.train:
        # Load and analyze data
        recommender.load_data(args.ratings_path, args.movie_list_path)
        
        # Train model
        history = recommender.train(args.ratings_path)
        
        # Save model if requested
        if args.save_model:
            print(f"\nSaving model to {args.save_model}...")
            recommender.save_model(args.save_model)
    
    # Generate recommendations if requested
    if args.recommend:
        if recommender.movie_list is None:
            recommender.load_data(args.ratings_path, args.movie_list_path)
        
        # Get ratings from user
        user_ratings = get_user_ratings(recommender.movie_list)
        
        if np.sum(user_ratings > 0) == 0:
            print("\nNo ratings provided. Cannot generate recommendations.")
            return
        
        # Generate and display recommendations
        print("\nGenerating recommendations...")
        recommendations = recommender.recommend_for_user(
            user_ratings, args.n_recommendations
        )
        
        print("\nTop Recommendations:")
        for _, row in recommendations.iterrows():
            print(f"{row['Movie']} (Predicted Rating: {row['Predicted Rating']:.1f})")

if __name__ == '__main__':
    main() 