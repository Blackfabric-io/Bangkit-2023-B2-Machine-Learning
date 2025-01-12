"""
Utility functions for data preprocessing, visualization, and model evaluation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, Any
import numpy.typing as npt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

ArrayType = npt.NDArray[np.float64]

def load_data(filepath: str) -> pd.DataFrame:
    """Load admission data from CSV file.
    
    Args:
        filepath: Path to the CSV file.
        
    Returns:
        DataFrame containing admission data.
        
    Raises:
        FileNotFoundError: If file doesn't exist.
    """
    try:
        return pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found at {filepath}")

def preprocess_data(df: pd.DataFrame, feature_cols: list, target_col: str, 
                   test_size: float = 0.2, random_state: int = 42) -> Tuple[ArrayType, ArrayType, ArrayType, ArrayType]:
    """Preprocess data for model training.
    
    Args:
        df: Input DataFrame.
        feature_cols: List of feature column names.
        target_col: Name of target column.
        test_size: Proportion of dataset to include in the test split.
        random_state: Random state for reproducibility.
        
    Returns:
        Tuple containing:
            X_train: Training features.
            X_test: Testing features.
            y_train: Training labels.
            y_test: Testing labels.
    """
    # Extract features and target
    X = df[feature_cols].values
    y = df[target_col].values
    
    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Split data
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def plot_feature_distributions(df: pd.DataFrame, feature_cols: list, target_col: str) -> None:
    """Plot feature distributions by class.
    
    Args:
        df: Input DataFrame.
        feature_cols: List of feature column names.
        target_col: Name of target column.
    """
    n_features = len(feature_cols)
    fig, axes = plt.subplots(2, (n_features + 1) // 2, figsize=(15, 10))
    axes = axes.ravel()
    
    for idx, feature in enumerate(feature_cols):
        sns.boxplot(x=target_col, y=feature, data=df, ax=axes[idx])
        axes[idx].set_title(f'Distribution of {feature}')
    
    plt.tight_layout()
    plt.show()

def plot_correlation_matrix(df: pd.DataFrame, feature_cols: list) -> None:
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

def plot_decision_boundary(X: ArrayType, y: ArrayType, model: Any, 
                         feature_idx: Tuple[int, int] = (0, 1)) -> None:
    """Plot decision boundary for two selected features.
    
    Args:
        X: Feature array.
        y: Target array.
        model: Fitted model instance.
        feature_idx: Tuple of feature indices to plot.
    """
    # Create mesh grid
    h = 0.02  # Step size
    x_min, x_max = X[:, feature_idx[0]].min() - 1, X[:, feature_idx[0]].max() + 1
    y_min, y_max = X[:, feature_idx[1]].min() - 1, X[:, feature_idx[1]].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # Create feature matrix for prediction
    X_mesh = np.zeros((xx.ravel().shape[0], X.shape[1]))
    X_mesh[:, feature_idx[0]] = xx.ravel()
    X_mesh[:, feature_idx[1]] = yy.ravel()
    
    # Predict and reshape
    Z = model.predict(X_mesh)
    Z = Z.reshape(xx.shape)
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, feature_idx[0]], X[:, feature_idx[1]], c=y, alpha=0.8)
    plt.xlabel(f'Feature {feature_idx[0]}')
    plt.ylabel(f'Feature {feature_idx[1]}')
    plt.title('Decision Boundary')
    plt.show()

def plot_cost_history(cost_history: list, title: str = "Cost History") -> None:
    """Plot training cost history.
    
    Args:
        cost_history: List of cost values during training.
        title: Plot title.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(cost_history)
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.show()

def evaluate_model(y_true: ArrayType, y_pred: ArrayType) -> Dict[str, Any]:
    """Evaluate model performance.
    
    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        
    Returns:
        Dictionary containing evaluation metrics.
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Compute metrics
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * (precision * recall) / (precision + recall)
    
    return {
        'confusion_matrix': cm,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'classification_report': classification_report(y_true, y_pred)
    } 