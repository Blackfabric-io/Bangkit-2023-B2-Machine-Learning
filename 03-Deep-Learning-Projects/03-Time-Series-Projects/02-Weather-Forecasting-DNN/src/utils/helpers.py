"""Helper functions for data processing and visualization."""

from typing import Tuple, Dict, Optional
import numpy as np
import numpy.typing as npt
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

def load_weather_data(csv_path: str) -> pd.DataFrame:
    """Load weather data from CSV file.
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        DataFrame containing weather data
        
    Raises:
        FileNotFoundError: If CSV file doesn't exist
    """
    try:
        df = pd.read_csv(csv_path)
        df['Date Time'] = pd.to_datetime(df['Date Time'])
        df = df.set_index('Date Time')
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"Weather data file not found at: {csv_path}")

def normalize_data(data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Tuple[float, float]]]:
    """Normalize data to [0, 1] range.
    
    Args:
        data: Input DataFrame
        
    Returns:
        Tuple containing:
            - Normalized DataFrame
            - Dictionary of (min, max) values for each column
    """
    result = pd.DataFrame()
    feature_ranges = {}
    
    for column in data.columns:
        min_val = data[column].min()
        max_val = data[column].max()
        
        result[column] = (data[column] - min_val) / (max_val - min_val)
        feature_ranges[column] = (min_val, max_val)
        
    return result, feature_ranges

def denormalize_data(data: npt.NDArray[np.float32],
                    feature_ranges: Dict[str, Tuple[float, float]],
                    feature_names: Optional[list] = None) -> pd.DataFrame:
    """Denormalize data back to original scale.
    
    Args:
        data: Normalized data array
        feature_ranges: Dictionary of (min, max) values
        feature_names: List of feature names
        
    Returns:
        Denormalized DataFrame
    """
    if feature_names is None:
        feature_names = list(feature_ranges.keys())
        
    result = pd.DataFrame()
    
    for i, column in enumerate(feature_names):
        min_val, max_val = feature_ranges[column]
        result[column] = data[:, i] * (max_val - min_val) + min_val
        
    return result

def plot_features(features: pd.DataFrame,
                 title: str,
                 xlabel: str = 'Date',
                 ylabel: str = 'Value') -> None:
    """Plot multiple features over time.
    
    Args:
        features: DataFrame of features to plot
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
    """
    plt.figure(figsize=(12, 8))
    for column in features.columns:
        plt.plot(features.index, features[column], label=column)
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.show()

def create_sequences(data: npt.NDArray[np.float32],
                    window_size: int,
                    batch_size: int = 32,
                    shuffle: bool = True) -> tf.data.Dataset:
    """Create sequences for time series prediction.
    
    Args:
        data: Input data array
        window_size: Size of sliding window
        batch_size: Training batch size
        shuffle: Whether to shuffle the data
        
    Returns:
        TensorFlow dataset of sequences
    """
    return tf.keras.preprocessing.timeseries_dataset_from_array(
        data=data,
        targets=None,
        sequence_length=window_size,
        sequence_stride=1,
        shuffle=shuffle,
        batch_size=batch_size) 