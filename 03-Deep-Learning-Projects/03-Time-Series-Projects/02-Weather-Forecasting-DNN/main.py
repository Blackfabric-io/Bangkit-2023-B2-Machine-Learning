"""Weather forecasting using Deep Neural Networks."""

import argparse
import matplotlib.pyplot as plt
from src.processors.main import prepare_data, train_model, evaluate_model, plot_predictions

def main():
    """Run weather forecasting pipeline."""
    parser = argparse.ArgumentParser(description='Weather forecasting using DNNs')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to weather data CSV file')
    parser.add_argument('--train_split', type=float, default=0.8,
                       help='Fraction of data to use for training')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience')
    parser.add_argument('--max_epochs', type=int, default=100,
                       help='Maximum training epochs')
    parser.add_argument('--features', nargs='+', default=['Temperature'],
                       help='Features to predict')
    args = parser.parse_args()
    
    # Prepare data
    print("\nPreparing data...")
    train_ds, val_ds, feature_ranges = prepare_data(
        args.data_path,
        args.train_split
    )
    
    # Train model
    print("\nTraining model...")
    model, history = train_model(
        train_ds,
        val_ds,
        num_features=len(args.features),
        patience=args.patience,
        max_epochs=args.max_epochs
    )
    
    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Evaluate model
    print("\nEvaluating model...")
    true_df, pred_df = evaluate_model(
        model,
        val_ds,
        feature_ranges,
        args.features
    )
    
    # Plot predictions for each feature
    for feature in args.features:
        plot_predictions(true_df, pred_df, feature)
        
    # Print evaluation metrics
    for feature in args.features:
        mae = abs(true_df[feature] - pred_df[feature]).mean()
        mse = ((true_df[feature] - pred_df[feature]) ** 2).mean()
        print(f"\nMetrics for {feature}:")
        print(f"MAE: {mae:.2f}")
        print(f"MSE: {mse:.2f}")

if __name__ == "__main__":
    main() 