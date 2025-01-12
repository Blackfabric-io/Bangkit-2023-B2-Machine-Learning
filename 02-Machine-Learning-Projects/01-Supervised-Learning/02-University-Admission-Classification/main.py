"""
Main script for training and evaluating the university admission classifier.
"""

import argparse
import json
from pathlib import Path
from src.processors.main import AdmissionClassifier

def main(args):
    """Main function to train and evaluate the model.
    
    Args:
        args: Command line arguments.
    """
    # Initialize classifier
    classifier = AdmissionClassifier(
        learning_rate=args.learning_rate,
        n_iterations=args.n_iterations
    )
    
    # Define features
    feature_cols = [
        'GRE Score',
        'TOEFL Score',
        'University Rating',
        'SOP',
        'LOR',
        'CGPA',
        'Research'
    ]
    
    # Train and evaluate
    print("Training model...")
    results = classifier.train(
        data_path=args.data_path,
        feature_cols=feature_cols,
        target_col='Admission',
        test_size=args.test_size,
        random_state=args.random_state,
        plot_training=args.plot_training
    )
    
    # Print results
    print("\nTraining Metrics:")
    print(f"Accuracy: {results['train_metrics']['accuracy']:.4f}")
    print(f"Precision: {results['train_metrics']['precision']:.4f}")
    print(f"Recall: {results['train_metrics']['recall']:.4f}")
    print(f"F1 Score: {results['train_metrics']['f1_score']:.4f}")
    
    print("\nTest Metrics:")
    print(f"Accuracy: {results['test_metrics']['accuracy']:.4f}")
    print(f"Precision: {results['test_metrics']['precision']:.4f}")
    print(f"Recall: {results['test_metrics']['recall']:.4f}")
    print(f"F1 Score: {results['test_metrics']['f1_score']:.4f}")
    
    print("\nFeature Importance:")
    for feature, importance in sorted(
        results['feature_importance'].items(),
        key=lambda x: x[1],
        reverse=True
    ):
        print(f"{feature}: {importance:.4f}")
    
    # Save results if requested
    if args.save_results:
        output_path = Path(args.save_results)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy values to Python types for JSON serialization
        results = {
            'train_metrics': {
                k: float(v) if isinstance(v, (np.float32, np.float64)) else v
                for k, v in results['train_metrics'].items()
                if k != 'classification_report'
            },
            'test_metrics': {
                k: float(v) if isinstance(v, (np.float32, np.float64)) else v
                for k, v in results['test_metrics'].items()
                if k != 'classification_report'
            },
            'feature_importance': {
                k: float(v) for k, v in results['feature_importance'].items()
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and evaluate university admission classifier"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the admission data CSV file"
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
        default=1000,
        help="Number of training iterations"
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Proportion of dataset to include in test split"
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random state for reproducibility"
    )
    parser.add_argument(
        "--plot_training",
        action="store_true",
        help="Plot training visualizations"
    )
    parser.add_argument(
        "--save_results",
        type=str,
        help="Path to save results JSON file"
    )
    
    args = parser.parse_args()
    main(args) 