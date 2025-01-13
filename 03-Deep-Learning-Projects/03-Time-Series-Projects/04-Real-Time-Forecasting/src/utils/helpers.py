"""Utility functions for data processing and visualization."""

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List, Dict, Union, Optional

def prepare_data(
    df: pd.DataFrame,
    input_width: int = 24,
    label_width: int = 1,
    shift: int = 1,
    train_split: float = 0.8,
    batch_size: int = 32,
    shuffle: bool = True
) -> Tuple[tf.data.Dataset, tf.data.Dataset, StandardScaler]:
    """Prepare data for model training.
    
    Args:
        df: Input DataFrame with timestamp index
        input_width: Number of input time steps
        label_width: Number of output time steps
        shift: Number of time steps to shift the target
        train_split: Fraction of data to use for training
        batch_size: Batch size for training
        shuffle: Whether to shuffle the training data
        
    Returns:
        train_ds: Training dataset
        val_ds: Validation dataset
        scaler: Fitted StandardScaler
    """
    # Scale the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df.values)
    
    # Create windows
    windows = create_windows(scaled_data, input_width + shift, stride=1)
    
    # Split into inputs and labels
    inputs = windows[:, :input_width, :]
    labels = windows[:, -label_width:, :]
    
    # Split into train and validation
    n_train = int(len(inputs) * train_split)
    train_inputs, train_labels = inputs[:n_train], labels[:n_train]
    val_inputs, val_labels = inputs[n_train:], labels[n_train:]
    
    # Create TensorFlow datasets
    train_ds = tf.data.Dataset.from_tensor_slices((train_inputs, train_labels))
    if shuffle:
        train_ds = train_ds.shuffle(buffer_size=len(train_inputs))
    train_ds = train_ds.batch(batch_size)
    
    val_ds = tf.data.Dataset.from_tensor_slices((val_inputs, val_labels))
    val_ds = val_ds.batch(batch_size)
    
    return train_ds, val_ds, scaler

def create_windows(
    data: np.ndarray,
    window_size: int,
    stride: int = 1
) -> np.ndarray:
    """Create sliding windows from data.
    
    Args:
        data: Input array of shape (timesteps, features)
        window_size: Size of the sliding window
        stride: Step size between windows
        
    Returns:
        windows: Array of shape (n_windows, window_size, features)
    """
    n_samples = (len(data) - window_size) // stride + 1
    n_features = data.shape[1]
    
    windows = np.zeros((n_samples, window_size, n_features))
    for i in range(n_samples):
        start_idx = i * stride
        end_idx = start_idx + window_size
        windows[i] = data[start_idx:end_idx]
    
    return windows

def plot_training_history(
    history: Dict[str, List[float]],
    metrics: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> None:
    """Plot training history.
    
    Args:
        history: Training history dictionary
        metrics: List of metrics to plot (default: ['loss'])
        figsize: Figure size
    """
    if metrics is None:
        metrics = ['loss']
    
    plt.figure(figsize=figsize)
    for metric in metrics:
        plt.plot(history[metric], label=f'Training {metric}')
        if f'val_{metric}' in history:
            plt.plot(history[f'val_{metric}'], label=f'Validation {metric}')
    
    plt.title('Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Metric Value')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_predictions(
    true_values: np.ndarray,
    predictions: np.ndarray,
    std: Optional[np.ndarray] = None,
    feature_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (15, 5 * 3),
    max_samples: int = 100
) -> None:
    """Plot true values vs predictions with optional uncertainty.
    
    Args:
        true_values: True values array
        predictions: Predicted values array
        std: Standard deviation of predictions (optional)
        feature_names: List of feature names (optional)
        figsize: Figure size
        max_samples: Maximum number of samples to plot
    """
    n_features = true_values.shape[-1]
    if feature_names is None:
        feature_names = [f'Feature {i+1}' for i in range(n_features)]
    
    # Limit number of samples
    if len(true_values) > max_samples:
        indices = np.linspace(0, len(true_values)-1, max_samples, dtype=int)
        true_values = true_values[indices]
        predictions = predictions[indices]
        if std is not None:
            std = std[indices]
    
    plt.figure(figsize=figsize)
    for i in range(n_features):
        plt.subplot(n_features, 1, i+1)
        plt.plot(true_values[:, 0, i], label='True', marker='o')
        plt.plot(predictions[:, 0, i], label='Predicted', marker='s')
        
        if std is not None:
            plt.fill_between(
                range(len(predictions)),
                predictions[:, 0, i] - 2*std[:, 0, i],
                predictions[:, 0, i] + 2*std[:, 0, i],
                alpha=0.2,
                label='95% CI'
            )
        
        plt.title(f'{feature_names[i]} Predictions')
        plt.xlabel('Time Step')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def compute_metrics(
    true_values: np.ndarray,
    predictions: np.ndarray,
    feature_names: Optional[List[str]] = None
) -> pd.DataFrame:
    """Compute evaluation metrics.
    
    Args:
        true_values: True values array
        predictions: Predicted values array
        feature_names: List of feature names (optional)
        
    Returns:
        metrics_df: DataFrame with computed metrics
    """
    n_features = true_values.shape[-1]
    if feature_names is None:
        feature_names = [f'Feature {i+1}' for i in range(n_features)]
    
    metrics = {
        'MSE': [],
        'MAE': [],
        'RMSE': [],
        'MAPE': []
    }
    
    for i in range(n_features):
        true = true_values[:, 0, i]
        pred = predictions[:, 0, i]
        
        mse = np.mean((true - pred) ** 2)
        mae = np.mean(np.abs(true - pred))
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((true - pred) / true)) * 100
        
        metrics['MSE'].append(mse)
        metrics['MAE'].append(mae)
        metrics['RMSE'].append(rmse)
        metrics['MAPE'].append(mape)
    
    metrics_df = pd.DataFrame(metrics, index=feature_names)
    return metrics_df

def exponential_smoothing(
    data: np.ndarray,
    alpha: float = 0.3
) -> np.ndarray:
    """Apply exponential smoothing to time series data.
    
    Args:
        data: Input array of shape (timesteps, features)
        alpha: Smoothing factor (0 < alpha < 1)
        
    Returns:
        smoothed: Smoothed array
    """
    smoothed = np.zeros_like(data)
    smoothed[0] = data[0]
    
    for t in range(1, len(data)):
        smoothed[t] = alpha * data[t] + (1 - alpha) * smoothed[t-1]
    
    return smoothed 