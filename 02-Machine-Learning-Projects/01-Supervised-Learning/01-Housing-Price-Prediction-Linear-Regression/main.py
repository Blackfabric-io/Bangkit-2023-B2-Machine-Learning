"""
Main script for training and evaluating the housing price prediction model.
"""

import argparse
from pathlib import Path
from src.core.base import LinearRegression
from src.utils.helpers import (
    load_data, 
    preprocess_data, 
    plot_data, 
    plot_regression_line, 
    plot_cost_history,
    compute_metrics
)

def main(args):
    """Main function to train and evaluate the model.
    
    Args:
        args: Command line arguments.
    """
    # Load and preprocess data
    print("Loading data...")
    df = load_data(args.data_path)
    X, y = preprocess_data(df, args.feature_col, args.target_col)
    
    if args.plot_data:
        plot_data(X, y)
    
    # Initialize and train model
    print("\nTraining model...")
    model = LinearRegression(
        learning_rate=args.learning_rate,
        n_iterations=args.n_iterations
    )
    model.fit(X, y)
    
    if args.plot_cost:
        plot_cost_history(model.cost_history)
    
    # Make predictions and evaluate
    print("\nEvaluating model...")
    y_pred = model.predict(X)
    metrics = compute_metrics(y, y_pred)
    
    print(f"\nModel Performance Metrics:")
    print(f"MSE: {metrics['mse']:.2f}")
    print(f"R² Score: {metrics['r2_score']:.2f}")
    print(f"MAPE: {metrics['mape']:.2f}%")
    
    if args.plot_fit:
        plot_regression_line(X, y, model.w, model.b)
    
    print(f"\nFinal model parameters:")
    print(f"Weight (w): {model.w:.4f}")
    print(f"Bias (b): {model.b:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and evaluate housing price prediction model"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the housing data CSV file"
    )
    parser.add_argument(
        "--feature_col",
        type=str,
        default="square_feet",
        help="Name of the feature column"
    )
    parser.add_argument(
        "--target_col",
        type=str,
        default="price",
        help="Name of the target column"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.01,
        help="Learning rate for gradient descent"
    )
    parser.add_argument(
        "--n_iterations",
        type=int,
        default=1500,
        help="Number of training iterations"
    )
    parser.add_argument(
        "--plot_data",
        action="store_true",
        help="Plot training data"
    )
    parser.add_argument(
        "--plot_cost",
        action="store_true",
        help="Plot cost history"
    )
    parser.add_argument(
        "--plot_fit",
        action="store_true",
        help="Plot regression line fit"
    )
    
    args = parser.parse_args()
    main(args) 