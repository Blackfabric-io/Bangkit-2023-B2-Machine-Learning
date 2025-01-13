"""Helper functions for data processing and visualization."""

from typing import Tuple, Optional
import numpy as np
import numpy.typing as npt
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def load_stock_data(csv_path: str) -> pd.DataFrame:
    """Load stock price data from CSV file.
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        DataFrame containing stock data
        
    Raises:
        FileNotFoundError: If CSV file doesn't exist
    """
    try:
        df = pd.read_csv(csv_path)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"Stock data file not found at: {csv_path}")

def prepare_data(data: pd.DataFrame,
                sequence_length: int = 60,
                train_split: float = 0.8,
                target_column: str = 'Close') -> Tuple[npt.NDArray[np.float32],
                                                     npt.NDArray[np.float32],
                                                     npt.NDArray[np.float32],
                                                     npt.NDArray[np.float32],
                                                     MinMaxScaler]:
    """Prepare data for model training.
    
    Args:
        data: Input DataFrame
        sequence_length: Number of time steps in each sequence
        train_split: Fraction of data to use for training
        target_column: Column to predict
        
    Returns:
        Tuple containing:
            - Training features
            - Training targets
            - Validation features
            - Validation targets
            - Data scaler
    """
    # Extract target values
    values = data[target_column].values.reshape(-1, 1)
    
    # Scale data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(values)
    
    # Create sequences
    X, y = create_sequences(scaled_data, sequence_length)
    
    # Split into train/validation
    train_size = int(len(X) * train_split)
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_val = X[train_size:]
    y_val = y[train_size:]
    
    return X_train, y_train, X_val, y_val, scaler

def create_sequences(data: npt.NDArray[np.float32],
                    sequence_length: int) -> Tuple[npt.NDArray[np.float32],
                                                 npt.NDArray[np.float32]]:
    """Create sequences for time series prediction.
    
    Args:
        data: Input data array
        sequence_length: Number of time steps in each sequence
        
    Returns:
        Tuple containing:
            - Input sequences
            - Target values
    """
    X = []
    y = []
    
    for i in range(len(data) - sequence_length):
        X.append(data[i:(i + sequence_length)])
        y.append(data[i + sequence_length])
        
    return np.array(X), np.array(y)

def plot_predictions(true_values: npt.NDArray[np.float32],
                    predicted_values: npt.NDArray[np.float32],
                    title: str = "Stock Price Prediction",
                    start_idx: Optional[int] = None) -> None:
    """Plot true vs predicted values.
    
    Args:
        true_values: Ground truth values
        predicted_values: Model predictions
        title: Plot title
        start_idx: Starting index for plotting
    """
    plt.figure(figsize=(12, 6))
    
    if start_idx is not None:
        true_values = true_values[start_idx:]
        predicted_values = predicted_values[start_idx:]
    
    plt.plot(true_values, label='True')
    plt.plot(predicted_values, label='Predicted')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()

def evaluate_predictions(true_values: npt.NDArray[np.float32],
                        predicted_values: npt.NDArray[np.float32]) -> Tuple[float, float]:
    """Compute evaluation metrics.
    
    Args:
        true_values: Ground truth values
        predicted_values: Model predictions
        
    Returns:
        Tuple containing:
            - Mean squared error
            - Mean absolute error
    """
    mse = np.mean((true_values - predicted_values) ** 2)
    mae = np.mean(np.abs(true_values - predicted_values))
    return mse, mae 