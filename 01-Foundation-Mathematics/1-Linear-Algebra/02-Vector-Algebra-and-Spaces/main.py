"""
Main entry for neural network training and prediction.
"""

import argparse
import numpy as np
from src.processors.main import nn_model, normalize_data, predict
from src.utils.helpers import generate_regression_data, load_house_prices_data, plot_regression_line

def main():
    parser = argparse.ArgumentParser(description='Train and evaluate neural network for regression')
    parser.add_argument('--mode', type=str, choices=['simple', 'multiple'], required=True,
                      help='Mode: simple (one feature) or multiple (two features) regression')
    parser.add_argument('--data', type=str, choices=['synthetic', 'house_prices'],
                      default='synthetic', help='Dataset to use')
    parser.add_argument('--iterations', type=int, default=100,
                      help='Number of training iterations')
    parser.add_argument('--plot', action='store_true',
                      help='Plot results (only for simple regression)')
    parser.add_argument('--predict', type=str,
                      help='Comma-separated values to predict on')
    
    args = parser.parse_args()
    
    try:
        # Load data
        if args.data == 'synthetic':
            if args.mode == 'simple':
                X, Y = generate_regression_data(n_features=1)
            else:
                X, Y = generate_regression_data(n_features=2)
        else:
            X, Y = load_house_prices_data('data/house_prices_train.csv')
            
        # Normalize data
        X_norm, X_mean, X_std = normalize_data(X)
        Y_norm, Y_mean, Y_std = normalize_data(Y)
        
        # Train model
        print(f"\nTraining neural network for {args.mode} regression...")
        parameters = nn_model(X_norm, Y_norm, num_iterations=args.iterations, print_cost=True)
        print("\nTraining completed!")
        
        # Make predictions if requested
        if args.predict:
            try:
                X_pred = np.array([float(x) for x in args.predict.split(',')]).reshape(-1, 1)
                if args.mode == 'multiple' and X_pred.shape[0] != 2:
                    raise ValueError("For multiple regression, provide 2 comma-separated values")
                    
                Y_pred = predict(X_pred, parameters, X_mean, X_std, Y_mean[0,0], Y_std[0,0])
                print(f"\nPredictions for input {args.predict}:")
                print(f"Predicted value: {Y_pred[0,0]:.2f}")
                
                if args.plot and args.mode == 'simple':
                    plot_regression_line(X, Y, parameters, X_pred)
                    
            except ValueError as e:
                print(f"Error making predictions: {str(e)}")
                return 1
                
        # Plot results for simple regression
        elif args.plot and args.mode == 'simple':
            plot_regression_line(X, Y, parameters)
            
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
        
    return 0

if __name__ == "__main__":
    exit(main()) 