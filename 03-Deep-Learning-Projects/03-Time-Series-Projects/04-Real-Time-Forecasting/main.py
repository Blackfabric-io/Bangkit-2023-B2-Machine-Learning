"""Main execution script for real-time forecasting."""

import argparse
import logging
import os
import pandas as pd
from src.processors.main import train_model, evaluate_model, simulate_real_time
from src.utils.helpers import plot_training_history, plot_predictions

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Real-time forecasting with deep learning'
    )
    
    # Data parameters
    parser.add_argument(
        '--data_path',
        type=str,
        required=True,
        help='Path to input data CSV file'
    )
    parser.add_argument(
        '--input_width',
        type=int,
        default=24,
        help='Number of input time steps'
    )
    parser.add_argument(
        '--label_width',
        type=int,
        default=1,
        help='Number of output time steps'
    )
    parser.add_argument(
        '--shift',
        type=int,
        default=1,
        help='Number of time steps to shift target'
    )
    
    # Training parameters
    parser.add_argument(
        '--train_split',
        type=float,
        default=0.8,
        help='Fraction of data for training'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Training batch size'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Maximum number of training epochs'
    )
    parser.add_argument(
        '--patience',
        type=int,
        default=10,
        help='Early stopping patience'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.001,
        help='Learning rate for optimization'
    )
    
    # Model parameters
    parser.add_argument(
        '--lstm_units',
        type=int,
        nargs='+',
        default=[64, 32],
        help='List of LSTM layer sizes'
    )
    parser.add_argument(
        '--dropout_rate',
        type=float,
        default=0.2,
        help='Dropout rate for regularization'
    )
    parser.add_argument(
        '--no_residual',
        action='store_true',
        help='Disable residual connections'
    )
    parser.add_argument(
        '--no_attention',
        action='store_true',
        help='Disable attention mechanism'
    )
    
    # Simulation parameters
    parser.add_argument(
        '--simulate',
        action='store_true',
        help='Run real-time simulation after training'
    )
    parser.add_argument(
        '--n_steps',
        type=int,
        default=100,
        help='Number of simulation steps'
    )
    parser.add_argument(
        '--smooth_predictions',
        action='store_true',
        help='Apply exponential smoothing to predictions'
    )
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.3,
        help='Smoothing factor for exponential smoothing'
    )
    
    # Output parameters
    parser.add_argument(
        '--model_dir',
        type=str,
        default='models',
        help='Directory to save trained model'
    )
    parser.add_argument(
        '--plot_results',
        action='store_true',
        help='Plot training history and predictions'
    )
    
    return parser.parse_args()

def main():
    """Main execution function."""
    # Set up logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Parse arguments
    args = parse_args()
    
    # Create model directory if needed
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    
    # Train model
    logger.info("Training model...")
    results = train_model(
        data_path=args.data_path,
        input_width=args.input_width,
        label_width=args.label_width,
        shift=args.shift,
        train_split=args.train_split,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        plot_history=args.plot_results
    )
    
    model = results['model']
    scaler = results['scaler']
    val_ds = results['validation_data']
    
    # Save model
    model_path = os.path.join(args.model_dir, 'model.h5')
    logger.info(f"Saving model to {model_path}")
    model.save(model_path)
    
    # Evaluate model
    logger.info("Evaluating model...")
    eval_results = evaluate_model(
        model=model,
        test_ds=val_ds,
        scaler=scaler,
        return_uncertainty=True,
        plot_results=args.plot_results
    )
    
    logger.info("\nEvaluation metrics:")
    print(eval_results['metrics'])
    
    # Run simulation if requested
    if args.simulate:
        logger.info("\nRunning real-time simulation...")
        df = pd.read_csv(args.data_path)
        if 'timestamp' in df.columns:
            df = df.set_index('timestamp')
        
        sim_results = simulate_real_time(
            model=model,
            data=df,
            scaler=scaler,
            window_size=args.input_width,
            n_steps=args.n_steps,
            smooth_predictions=args.smooth_predictions,
            alpha=args.alpha
        )
        
        logger.info("\nSimulation results:")
        print(sim_results.head())
        
        # Save simulation results
        sim_path = os.path.join(args.model_dir, 'simulation_results.csv')
        sim_results.to_csv(sim_path)
        logger.info(f"Saved simulation results to {sim_path}")

if __name__ == '__main__':
    main() 