"""
Utility functions for data preprocessing, visualization, and model evaluation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, List, Any
import numpy.typing as npt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

ArrayType = npt.NDArray[np.float64]

def load_data(filepath: str) -> pd.DataFrame:
    """Load transaction data from CSV file.
    
    Args:
        filepath: Path to the CSV file.
        
    Returns:
        DataFrame containing transaction data.
        
    Raises:
        FileNotFoundError: If file doesn't exist.
    """
    try:
        return pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found at {filepath}")

def preprocess_data(df: pd.DataFrame, feature_cols: List[str],
                   test_size: float = 0.2, random_state: int = 42
                   ) -> Tuple[ArrayType, ArrayType, StandardScaler]:
    """Preprocess transaction data for anomaly detection.
    
    Args:
        df: Input DataFrame.
        feature_cols: List of feature column names.
        test_size: Proportion of data to use for testing.
        random_state: Random state for reproducibility.
        
    Returns:
        Tuple containing:
            X_train: Training data.
            X_test: Test data.
            scaler: Fitted StandardScaler instance.
    """
    # Extract features
    X = df[feature_cols].values
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test = train_test_split(X_scaled, test_size=test_size,
                                      random_state=random_state)
    
    return X_train, X_test, scaler

def plot_feature_distributions(df: pd.DataFrame, feature_cols: List[str]) -> None:
    """Plot feature distributions.
    
    Args:
        df: Input DataFrame.
        feature_cols: List of feature column names.
    """
    n_features = len(feature_cols)
    n_rows = (n_features + 1) // 2
    
    plt.figure(figsize=(15, 5 * n_rows))
    for idx, feature in enumerate(feature_cols, 1):
        plt.subplot(n_rows, 2, idx)
        sns.histplot(data=df, x=feature, kde=True)
        plt.title(f'Distribution of {feature}')
    
    plt.tight_layout()
    plt.show()

def plot_correlation_matrix(df: pd.DataFrame, feature_cols: List[str]) -> None:
    """Plot correlation matrix of features.
    
    Args:
        df: Input DataFrame.
        feature_cols: List of feature column names.
    """
    corr = df[feature_cols].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Matrix')
    plt.show()

def plot_anomaly_scores(scores: ArrayType, threshold: float,
                       title: str = "Anomaly Scores Distribution") -> None:
    """Plot distribution of anomaly scores.
    
    Args:
        scores: Array of anomaly scores.
        threshold: Threshold for anomaly detection.
        title: Plot title.
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(scores, kde=True)
    plt.axvline(threshold, color='r', linestyle='--',
                label=f'Threshold: {threshold:.3f}')
    plt.title(title)
    plt.xlabel('Anomaly Score')
    plt.ylabel('Count')
    plt.legend()
    plt.show()

def plot_feature_importance(df: pd.DataFrame, feature_cols: List[str],
                          predictions: ArrayType) -> None:
    """Plot feature distributions for normal vs anomalous transactions.
    
    Args:
        df: Input DataFrame.
        feature_cols: List of feature column names.
        predictions: Array of predictions (-1 for anomalies, 1 for normal).
    """
    n_features = len(feature_cols)
    n_rows = (n_features + 1) // 2
    
    plt.figure(figsize=(15, 5 * n_rows))
    for idx, feature in enumerate(feature_cols, 1):
        plt.subplot(n_rows, 2, idx)
        sns.boxplot(x=predictions, y=df[feature])
        plt.title(f'{feature} Distribution by Class')
        plt.xticks([-1, 1], ['Anomaly', 'Normal'])
    
    plt.tight_layout()
    plt.show()

def compute_metrics(y_true: ArrayType, y_pred: ArrayType) -> Dict[str, float]:
    """Compute evaluation metrics for anomaly detection.
    
    Args:
        y_true: True labels (-1 for anomalies, 1 for normal).
        y_pred: Predicted labels (-1 for anomalies, 1 for normal).
        
    Returns:
        Dictionary containing evaluation metrics.
    """
    from sklearn.metrics import (accuracy_score, precision_score,
                               recall_score, f1_score)
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, pos_label=-1),
        'recall': recall_score(y_true, y_pred, pos_label=-1),
        'f1': f1_score(y_true, y_pred, pos_label=-1)
    }
    
    return metrics

def save_results(results: Dict[str, Any], filepath: str) -> None:
    """Save analysis results to JSON file.
    
    Args:
        results: Dictionary containing analysis results.
        filepath: Path to save results.
    """
    import json
    
    # Convert numpy values to Python types
    processed_results = {}
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            processed_results[key] = value.tolist()
        elif isinstance(value, np.float64):
            processed_results[key] = float(value)
        else:
            processed_results[key] = value
    
    with open(filepath, 'w') as f:
        json.dump(processed_results, f, indent=4) 