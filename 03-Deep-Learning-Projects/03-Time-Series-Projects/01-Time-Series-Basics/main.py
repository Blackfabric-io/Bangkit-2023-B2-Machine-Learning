"""Example usage of time series forecasting package."""

import argparse
import matplotlib.pyplot as plt
from src.utils.helpers import generate_time_series
from src.core.base import train_val_split, plot_series, compute_metrics
from src.processors.main import naive_forecast, moving_average_forecast, diff_forecast

def main():
    """Run example time series forecasting workflow."""
    parser = argparse.ArgumentParser(description='Time series forecasting example')
    parser.add_argument('--n_points', type=int, default=1461,
                       help='Number of time points')
    parser.add_argument('--split_time', type=int, default=1100,
                       help='Time step to split at')
    parser.add_argument('--window_size', type=int, default=30,
                       help='Window size for moving average')
    parser.add_argument('--period', type=int, default=365,
                       help='Seasonality period')
    args = parser.parse_args()
    
    # Generate synthetic data
    time, series = generate_time_series(n_points=args.n_points)
    
    # Split into training and validation
    time_train, series_train, time_valid, series_valid = train_val_split(
        time, series, args.split_time)
    
    # Plot training data
    plt.figure(figsize=(10, 6))
    plot_series(time_train, series_train, title="Training Data")
    plt.show()
    
    # Plot validation data
    plt.figure(figsize=(10, 6))
    plot_series(time_valid, series_valid, title="Validation Data")
    plt.show()
    
    # Compute and evaluate forecasts
    print("\nEvaluating forecasting methods:")
    
    # Naive forecast
    naive_pred = naive_forecast(series, args.split_time)
    mse, mae = compute_metrics(series_valid, naive_pred)
    print(f"\nNaive Forecast:")
    print(f"MSE: {mse:.2f}")
    print(f"MAE: {mae:.2f}")
    
    plt.figure(figsize=(10, 6))
    plot_series(time_valid, series_valid, label="Validation")
    plot_series(time_valid, naive_pred, label="Naive Forecast")
    plt.legend()
    plt.show()
    
    # Moving average forecast
    ma_pred = moving_average_forecast(series, args.window_size)
    ma_pred = ma_pred[args.split_time-args.window_size:]
    mse, mae = compute_metrics(series_valid, ma_pred)
    print(f"\nMoving Average Forecast:")
    print(f"MSE: {mse:.2f}")
    print(f"MAE: {mae:.2f}")
    
    plt.figure(figsize=(10, 6))
    plot_series(time_valid, series_valid, label="Validation")
    plot_series(time_valid, ma_pred, label="Moving Average")
    plt.legend()
    plt.show()
    
    # Differenced forecast
    diff_pred = diff_forecast(series, args.period, args.window_size, args.split_time)
    mse, mae = compute_metrics(series_valid, diff_pred)
    print(f"\nDifferenced Forecast:")
    print(f"MSE: {mse:.2f}")
    print(f"MAE: {mae:.2f}")
    
    plt.figure(figsize=(10, 6))
    plot_series(time_valid, series_valid, label="Validation")
    plot_series(time_valid, diff_pred, label="Differenced")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main() 