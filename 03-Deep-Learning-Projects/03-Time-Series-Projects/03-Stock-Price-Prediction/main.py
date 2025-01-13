"""Stock price prediction using RNN, LSTM, and BiLSTM models."""

import argparse
import matplotlib.pyplot as plt
from src.processors.main import train_model, evaluate_model, plot_training_history, compare_models
from src.core.base import RNNModel, LSTMModel, BiLSTMModel

def main():
    """Run stock price prediction pipeline."""
    parser = argparse.ArgumentParser(description='Stock price prediction using deep learning')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to stock data CSV file')
    parser.add_argument('--model', type=str, default='LSTM',
                       choices=['RNN', 'LSTM', 'BiLSTM'],
                       help='Model type to use')
    parser.add_argument('--sequence_length', type=int, default=60,
                       help='Number of time steps in each sequence')
    parser.add_argument('--train_split', type=float, default=0.8,
                       help='Fraction of data to use for training')
    parser.add_argument('--target_column', type=str, default='Close',
                       help='Column to predict')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--compare', action='store_true',
                       help='Compare all models')
    args = parser.parse_args()
    
    if args.compare:
        print("\nComparing all models...")
        results = compare_models(
            args.data_path,
            sequence_length=args.sequence_length,
            train_split=args.train_split,
            target_column=args.target_column
        )
        
        # Plot results for each model
        for name, model_results in results.items():
            print(f"\n{name} Results:")
            plot_training_history(model_results['training']['history'])
            
            evaluation = model_results['evaluation']
            print(f"MSE: {evaluation['mse']:.2f}")
            print(f"MAE: {evaluation['mae']:.2f}")
            
    else:
        # Select model class
        model_classes = {
            'RNN': RNNModel,
            'LSTM': LSTMModel,
            'BiLSTM': BiLSTMModel
        }
        model_class = model_classes[args.model]
        
        print(f"\nTraining {args.model} model...")
        
        # Train model
        training_results = train_model(
            model_class,
            args.data_path,
            sequence_length=args.sequence_length,
            train_split=args.train_split,
            target_column=args.target_column,
            epochs=args.epochs,
            patience=args.patience,
            learning_rate=args.learning_rate
        )
        
        # Plot training history
        plot_training_history(training_results['history'])
        
        # Evaluate model
        X_val, y_val = training_results['validation_data']
        evaluation_results = evaluate_model(
            training_results['model'],
            X_val,
            y_val,
            training_results['scaler']
        )
        
        print("\nModel Performance:")
        print(f"MSE: {evaluation_results['mse']:.2f}")
        print(f"MAE: {evaluation_results['mae']:.2f}")
        
        # Plot predictions
        plt.figure(figsize=(12, 6))
        plt.plot(y_val, label='True')
        plt.plot(evaluation_results['predictions'], label='Predicted')
        plt.title(f'{args.model} Model Predictions')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    main() 