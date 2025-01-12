"""
Main script for running fraud detection analysis.
"""

import argparse
from pathlib import Path
from src.processors.main import FraudDetector

def main(args):
    """Main function to run fraud detection.
    
    Args:
        args: Command line arguments.
    """
    # Define features
    feature_cols = [
        'amount',
        'time',
        'v1', 'v2', 'v3', 'v4', 'v5',
        'v6', 'v7', 'v8', 'v9', 'v10',
        'v11', 'v12', 'v13', 'v14', 'v15',
        'v16', 'v17', 'v18', 'v19', 'v20',
        'v21', 'v22', 'v23', 'v24', 'v25',
        'v26', 'v27', 'v28'
    ]
    
    # Initialize detector
    detector = FraudDetector(
        n_estimators=args.n_estimators,
        max_samples=args.max_samples,
        contamination=args.contamination,
        random_state=args.random_state
    )
    
    # Analyze data if requested
    if args.analyze_data:
        print("Analyzing transaction data...")
        detector.analyze_data(args.data_path, feature_cols)
    
    # Train model
    print("\nTraining fraud detection model...")
    X_train, X_test = detector.train(
        args.data_path,
        feature_cols,
        test_size=args.test_size,
        random_state=args.random_state
    )
    
    # Evaluate model
    print("\nEvaluating model...")
    results = detector.evaluate(X_test)
    
    # Analyze feature importance
    if args.analyze_features:
        print("\nAnalyzing feature importance...")
        detector.analyze_feature_importance(args.data_path)
    
    # Save model if requested
    if args.save_model:
        print("\nSaving model...")
        detector.save_model(args.save_model)
    
    # Save results if requested
    if args.save_results:
        from src.utils.helpers import save_results
        print("\nSaving results...")
        save_results(results, args.save_results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run fraud detection analysis"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the transaction data CSV file"
    )
    parser.add_argument(
        "--n_estimators",
        type=int,
        default=100,
        help="Number of isolation trees"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=256,
        help="Number of samples to draw for each tree"
    )
    parser.add_argument(
        "--contamination",
        type=float,
        default=0.1,
        help="Expected proportion of anomalies"
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Proportion of data to use for testing"
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random state for reproducibility"
    )
    parser.add_argument(
        "--analyze_data",
        action="store_true",
        help="Analyze data before training"
    )
    parser.add_argument(
        "--analyze_features",
        action="store_true",
        help="Analyze feature importance after training"
    )
    parser.add_argument(
        "--save_model",
        type=str,
        help="Directory to save trained model"
    )
    parser.add_argument(
        "--save_results",
        type=str,
        help="Path to save results JSON file"
    )
    
    args = parser.parse_args()
    main(args) 