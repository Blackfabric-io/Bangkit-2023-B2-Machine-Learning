"""Main processor module for model training and evaluation."""

import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Dict, List, Optional, Tuple, Union
from sklearn.preprocessing import StandardScaler
from src.core.base import RealTimeModel
from src.utils.helpers import (
    prepare_data,
    plot_training_history,
    plot_predictions,
    compute_metrics,
    exponential_smoothing
)

def train_model(
    data_path: str,
    input_width: int = 24,
    label_width: int = 1,
    shift: int = 1,
    train_split: float = 0.8,
    batch_size: int = 32,
    epochs: int = 100,
    patience: int = 10,
    learning_rate: float = 0.001,
    validation_split: float = 0.2,
    plot_history: bool = True
) -> Dict:
    """Train a real-time forecasting model.
    
    Args:
        data_path: Path to CSV data file
        input_width: Number of input time steps
        label_width: Number of output time steps
        shift: Number of time steps to shift the target
        train_split: Fraction of data to use for training
        batch_size: Training batch size
        epochs: Maximum number of training epochs
        patience: Early stopping patience
        learning_rate: Learning rate for optimization
        validation_split: Fraction of training data to use for validation
        plot_history: Whether to plot training history
        
    Returns:
        Dictionary containing trained model and training results
    """
    # Load and prepare data
    df = pd.read_csv(data_path)
    if 'timestamp' in df.columns:
        df = df.set_index('timestamp')
    
    train_ds, val_ds, scaler = prepare_data(
        df,
        input_width=input_width,
        label_width=label_width,
        shift=shift,
        train_split=train_split,
        batch_size=batch_size
    )
    
    # Initialize and compile model
    model = RealTimeModel(
        input_width=input_width,
        label_width=label_width,
        shift=shift,
        learning_rate=learning_rate
    )
    model.build_model(num_features=df.shape[1])
    model.compile_model()
    
    # Set up callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=patience//2,
            min_lr=1e-6
        )
    ]
    
    # Train model
    history = model.model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    if plot_history:
        plot_training_history(history.history)
    
    return {
        'model': model,
        'history': history.history,
        'scaler': scaler,
        'validation_data': val_ds
    }

def evaluate_model(
    model: RealTimeModel,
    test_ds: tf.data.Dataset,
    scaler: StandardScaler,
    feature_names: Optional[List[str]] = None,
    return_uncertainty: bool = True,
    mc_samples: int = 100,
    plot_results: bool = True
) -> Dict:
    """Evaluate model performance on test data.
    
    Args:
        model: Trained RealTimeModel
        test_ds: Test dataset
        scaler: Fitted StandardScaler
        feature_names: List of feature names
        return_uncertainty: Whether to compute prediction uncertainty
        mc_samples: Number of Monte Carlo samples for uncertainty
        plot_results: Whether to plot evaluation results
        
    Returns:
        Dictionary containing evaluation results
    """
    # Get all test data
    X_test, y_test = [], []
    for x, y in test_ds:
        X_test.append(x.numpy())
        y_test.append(y.numpy())
    X_test = np.concatenate(X_test)
    y_test = np.concatenate(y_test)
    
    # Make predictions
    if return_uncertainty:
        # Enable dropout for uncertainty estimation
        for layer in model.model.layers:
            if isinstance(layer, tf.keras.layers.Dropout):
                layer.training = True
        
        # Generate multiple predictions
        predictions = []
        for _ in range(mc_samples):
            pred = model.model.predict(X_test, verbose=0)
            predictions.append(pred)
        
        # Calculate mean and std
        predictions = np.array(predictions)
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
    else:
        mean_pred = model.model.predict(X_test, verbose=0)
        std_pred = None
    
    # Inverse transform predictions
    y_true = scaler.inverse_transform(y_test.reshape(-1, y_test.shape[-1]))
    y_pred = scaler.inverse_transform(mean_pred.reshape(-1, mean_pred.shape[-1]))
    if std_pred is not None:
        std_pred = std_pred.reshape(-1, std_pred.shape[-1]) * np.sqrt(scaler.var_)
    
    # Reshape back to 3D
    y_true = y_true.reshape(y_test.shape)
    y_pred = y_pred.reshape(mean_pred.shape)
    if std_pred is not None:
        std_pred = std_pred.reshape(mean_pred.shape)
    
    # Compute metrics
    metrics = compute_metrics(y_true, y_pred, feature_names)
    
    if plot_results:
        plot_predictions(y_true, y_pred, std_pred, feature_names)
    
    return {
        'predictions': y_pred,
        'true_values': y_true,
        'std': std_pred,
        'metrics': metrics
    }

def simulate_real_time(
    model: RealTimeModel,
    data: pd.DataFrame,
    scaler: StandardScaler,
    window_size: int = 24,
    n_steps: int = 100,
    smooth_predictions: bool = True,
    alpha: float = 0.3
) -> pd.DataFrame:
    """Simulate real-time forecasting.
    
    Args:
        model: Trained RealTimeModel
        data: Input DataFrame
        scaler: Fitted StandardScaler
        window_size: Size of sliding window
        n_steps: Number of simulation steps
        smooth_predictions: Whether to apply exponential smoothing
        alpha: Smoothing factor
        
    Returns:
        DataFrame containing simulation results
    """
    # Initialize results storage
    results = []
    scaled_data = scaler.transform(data.values)
    
    # Run simulation
    for step in range(n_steps):
        # Get current window
        start_idx = step
        end_idx = start_idx + window_size
        window = scaled_data[start_idx:end_idx]
        
        # Make prediction
        X = window.reshape(1, window_size, -1)
        y_pred = model.model.predict(X, verbose=0)
        
        # Get true value
        y_true = scaled_data[end_idx:end_idx+1]
        
        # Compute loss
        loss = np.mean((y_true - y_pred) ** 2)
        
        # Store results
        results.append({
            'step': step,
            'true': y_true[0],
            'predicted': y_pred[0, 0],
            'loss': loss
        })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Inverse transform predictions
    for col in ['true', 'predicted']:
        values = np.array([r[col] for r in results])
        values = scaler.inverse_transform(values)
        results_df[col] = values.tolist()
    
    # Apply smoothing if requested
    if smooth_predictions:
        smoothed = []
        for feature in range(data.shape[1]):
            feature_preds = np.array([p[feature] for p in results_df['predicted']])
            smoothed_preds = exponential_smoothing(feature_preds, alpha)
            smoothed.append(smoothed_preds)
        results_df['smoothed'] = list(zip(*smoothed))
    
    return results_df 