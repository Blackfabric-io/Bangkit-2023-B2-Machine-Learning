"""Helper functions for data loading and visualization."""

import os
from typing import Tuple, List, Dict, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix

def load_data(data_dir: str) -> Tuple[str, str]:
    """Load training and test data files.
    
    Args:
        data_dir: Directory containing data files
        
    Returns:
        Tuple of (train_file_path, test_file_path)
        
    Raises:
        FileNotFoundError: If data files don't exist
    """
    train_file = os.path.join(data_dir, 'bbc-text.csv')
    test_file = os.path.join(data_dir, 'bbc-text-test.csv')
    
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"Training file not found: {train_file}")
    if not os.path.exists(test_file):
        raise FileNotFoundError(f"Test file not found: {test_file}")
        
    return train_file, test_file

def plot_metrics(history: Dict[str, List[float]], 
                figsize: Tuple[int, int] = (12, 4)) -> None:
    """Plot training metrics.
    
    Args:
        history: Training history dictionary
        figsize: Figure size
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot accuracy
    ax1.plot(history['accuracy'], label='Training')
    if 'val_accuracy' in history:
        ax1.plot(history['val_accuracy'], label='Validation')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    # Plot loss
    ax2.plot(history['loss'], label='Training')
    if 'val_loss' in history:
        ax2.plot(history['val_loss'], label='Validation')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true: List[int], 
                         y_pred: List[int],
                         labels: List[str],
                         figsize: Tuple[int, int] = (10, 8)) -> None:
    """Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Label names
        figsize: Figure size
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=figsize)
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels
    )
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.show() 