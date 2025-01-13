"""Main weather forecasting model training and evaluation."""

from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd
import tensorflow as tf
from ..core.base import WeatherModel
from ..utils.helpers import (load_weather_data, normalize_data,
                           denormalize_data, plot_features)

def prepare_data(data_path: str,
                train_split: float = 0.8) -> Tuple[tf.data.Dataset,
                                                 tf.data.Dataset,
                                                 Dict[str, Tuple[float, float]]]:
    """Prepare data for model training.
    
    Args:
        data_path: Path to weather data CSV
        train_split: Fraction of data to use for training
        
    Returns:
        Tuple containing:
            - Training dataset
            - Validation dataset
            - Feature normalization ranges
    """
    # Load and normalize data
    df = load_weather_data(data_path)
    normalized_data, feature_ranges = normalize_data(df)
    
    # Split into train/validation
    n = len(normalized_data)
    train_df = normalized_data[0:int(n*train_split)]
    val_df = normalized_data[int(n*train_split):]
    
    # Convert to numpy arrays
    train_data = np.array(train_df)
    val_data = np.array(val_df)
    
    # Create model and datasets
    model = WeatherModel()
    train_ds = model.make_dataset(train_data)
    val_ds = model.make_dataset(val_data, shuffle=False)
    
    return train_ds, val_ds, feature_ranges

def train_model(train_ds: tf.data.Dataset,
                val_ds: tf.data.Dataset,
                num_features: int,
                patience: int = 10,
                max_epochs: int = 100) -> Tuple[WeatherModel, Dict[str, Any]]:
    """Train the weather forecasting model.
    
    Args:
        train_ds: Training dataset
        val_ds: Validation dataset
        num_features: Number of input features
        patience: Early stopping patience
        max_epochs: Maximum training epochs
        
    Returns:
        Tuple containing:
            - Trained model
            - Training history
    """
    model = WeatherModel()
    history = model.compile_and_fit(
        train_ds,
        val_ds,
        num_features,
        patience=patience,
        max_epochs=max_epochs
    )
    
    return model, history.history

def evaluate_model(model: WeatherModel,
                  val_ds: tf.data.Dataset,
                  feature_ranges: Dict[str, Tuple[float, float]],
                  feature_names: list) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate model performance.
    
    Args:
        model: Trained model
        val_ds: Validation dataset
        feature_ranges: Feature normalization ranges
        feature_names: Names of features
        
    Returns:
        Tuple containing:
            - True values DataFrame
            - Predicted values DataFrame
    """
    # Get predictions
    predictions = model.predict(val_ds)
    
    # Get true values from validation set
    true_values = np.concatenate([y for x, y in val_ds], axis=0)
    
    # Denormalize predictions and true values
    pred_df = denormalize_data(predictions, feature_ranges, feature_names)
    true_df = denormalize_data(true_values, feature_ranges, feature_names)
    
    return true_df, pred_df

def plot_predictions(true_df: pd.DataFrame,
                    pred_df: pd.DataFrame,
                    feature: str) -> None:
    """Plot true vs predicted values for a feature.
    
    Args:
        true_df: DataFrame of true values
        pred_df: DataFrame of predicted values
        feature: Feature to plot
    """
    plt.figure(figsize=(12, 8))
    plt.plot(true_df.index, true_df[feature], label='True')
    plt.plot(pred_df.index, pred_df[feature], label='Predicted')
    plt.title(f'{feature} - True vs Predicted')
    plt.xlabel('Date')
    plt.ylabel(feature)
    plt.legend()
    plt.grid(True)
    plt.show() 