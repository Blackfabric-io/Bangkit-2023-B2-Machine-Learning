"""
Main script for running customer segmentation analysis.
"""

import argparse
import json
from pathlib import Path
from src.core.base import InitMethod
from src.processors.main import CustomerSegmentation

def main(args):
    """Main function to run customer segmentation.
    
    Args:
        args: Command line arguments.
    """
    # Define features
    feature_cols = [
        'Annual Income (k$)',
        'Spending Score (1-100)',
        'Age',
        'Shopping Frequency',
        'Purchase Amount'
    ]
    
    # Initialize segmentation model
    segmentation = CustomerSegmentation(
        n_clusters=args.n_clusters,
        max_iters=args.max_iters,
        init_method=InitMethod(args.init_method),
        random_state=args.random_state
    )
    
    # Analyze data if requested
    if args.analyze_data:
        print("Analyzing customer data...")
        segmentation.analyze_data(args.data_path, feature_cols)
    
    # Fit model and get results
    print("\nSegmenting customers...")
    results = segmentation.fit(args.data_path, feature_cols)
    
    # Print segment profiles
    print("\nSegment Profiles:")
    for i in range(args.n_clusters):
        print(f"\nSegment {i}:")
        profile = segmentation.get_segment_profile(i)
        for feature, value in profile.items():
            print(f"{feature}: {value:.2f}")
    
    # Save results if requested
    if args.save_results:
        output_path = Path(args.save_results)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy values to Python types for JSON serialization
        results = {
            'n_segments': args.n_clusters,
            'total_customers': len(results['labels']),
            'inertia': float(results['inertia']),
            'n_iterations': int(results['n_iterations']),
            'segment_sizes': {
                f'segment_{i}': int(sum(results['labels'] == i))
                for i in range(args.n_clusters)
            },
            'segment_profiles': {
                f'segment_{i}': {
                    feature: float(value)
                    for feature, value in segmentation.get_segment_profile(i).items()
                }
                for i in range(args.n_clusters)
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run customer segmentation analysis"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the customer data CSV file"
    )
    parser.add_argument(
        "--n_clusters",
        type=int,
        default=3,
        help="Number of customer segments"
    )
    parser.add_argument(
        "--max_iters",
        type=int,
        default=300,
        help="Maximum number of iterations"
    )
    parser.add_argument(
        "--init_method",
        type=str,
        choices=["random", "kmeans++"],
        default="kmeans++",
        help="Method for initializing centroids"
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
        help="Analyze data before segmentation"
    )
    parser.add_argument(
        "--save_results",
        type=str,
        help="Path to save results JSON file"
    )
    
    args = parser.parse_args()
    main(args) 