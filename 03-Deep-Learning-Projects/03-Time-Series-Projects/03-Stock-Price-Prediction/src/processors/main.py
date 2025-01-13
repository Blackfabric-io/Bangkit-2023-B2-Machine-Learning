"""Main stock price prediction model training and evaluation."""

from typing import Dict, Any, Type
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from ..core.base import BaseModel, RNNModel, LSTMModel, BiLSTMModel
from ..utils.helpers import (load_stock_data, prepare_data,
                           plot_predictions, evaluate_predictions)

def train_model(model_class: Type[BaseModel],
                data_path: str,
                sequence_length: int = 60,
                train_split: float = 0.8,
                target_column: str = 'Close',
                epochs: int = 50,
                patience: int = 10,
                learning_rate: float = 0.001) -> Dict[str, Any]:
    """Train a stock price prediction model.
    
    Args:
        model_class: Model class to use
        data_path: Path to stock data CSV
        sequence_length: Number of time steps in each sequence
        train_split: Fraction of data to use for training
        target_column: Column to predict
        epochs: Number of training epochs
        patience: Early stopping patience
        learning_rate: Learning rate for optimizer
        
    Returns:
        Dictionary containing:
            - Trained model
            - Training history
            - Data scaler
            - Validation data
    """
    # Load and prepare data
    df = load_stock_data(data_path)
    X_train, y_train, X_val, y_val, scaler = prepare_data(
        df,
        sequence_length=sequence_length,
        train_split=train_split,
        target_column=target_column
    )
    
    # Initialize and train model
    model = model_class(
        sequence_length=sequence_length,
        n_features=1
    )
    model.compile_model(learning_rate=learning_rate)
    
    history = model.train(
        X_train, y_train,
        X_val, y_val,
        epochs=epochs,
        patience=patience
    )
    
    return {
        'model': model,
        'history': history,
        'scaler': scaler,
        'validation_data': (X_val, y_val)
    }

def evaluate_model(model: BaseModel,
                  X: np.ndarray,
                  y_true: np.ndarray,
                  scaler: MinMaxScaler) -> Dict[str, Any]:
    """Evaluate model performance.
    
    Args:
        model: Trained model
        X: Input features
        y_true: True values
        scaler: Data scaler
        
    Returns:
        Dictionary containing:
            - Predictions
            - MSE
            - MAE
    """
    # Make predictions
    y_pred = model.predict(X)
    
    # Inverse transform predictions and true values
    y_true_orig = scaler.inverse_transform(y_true)
    y_pred_orig = scaler.inverse_transform(y_pred)
    
    # Compute metrics
    mse, mae = evaluate_predictions(y_true_orig, y_pred_orig)
    
    return {
        'predictions': y_pred_orig,
        'mse': mse,
        'mae': mae
    }

def plot_training_history(history: Dict[str, Any]) -> None:
    """Plot model training history.
    
    Args:
        history: Training history dictionary
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

def compare_models(data_path: str,
                  sequence_length: int = 60,
                  train_split: float = 0.8,
                  target_column: str = 'Close') -> Dict[str, Dict[str, Any]]:
    """Train and compare different models.
    
    Args:
        data_path: Path to stock data CSV
        sequence_length: Number of time steps in each sequence
        train_split: Fraction of data to use for training
        target_column: Column to predict
        
    Returns:
        Dictionary containing results for each model
    """
    models = {
        'RNN': RNNModel,
        'LSTM': LSTMModel,
        'BiLSTM': BiLSTMModel
    }
    
    results = {}
    
    for name, model_class in models.items():
        print(f"\nTraining {name} model...")
        
        # Train model
        training_results = train_model(
            model_class,
            data_path,
            sequence_length=sequence_length,
            train_split=train_split,
            target_column=target_column
        )
        
        # Evaluate model
        X_val, y_val = training_results['validation_data']
        evaluation_results = evaluate_model(
            training_results['model'],
            X_val,
            y_val,
            training_results['scaler']
        )
        
        results[name] = {
            'training': training_results,
            'evaluation': evaluation_results
        }
        
        print(f"{name} MSE: {evaluation_results['mse']:.2f}")
        print(f"{name} MAE: {evaluation_results['mae']:.2f}")
    
    return results 