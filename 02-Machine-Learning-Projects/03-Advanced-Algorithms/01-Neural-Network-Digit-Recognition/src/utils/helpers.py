"""
Utility functions for digit recognition.
"""

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import gzip
import pickle
import urllib.request
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Type aliases
ArrayType = npt.NDArray[np.float64]

def download_mnist(data_dir: str = 'data') -> None:
    """Download MNIST dataset if not already present.
    
    Args:
        data_dir: Directory to store dataset.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    base_url = "http://yann.lecun.com/exdb/mnist/"
    files = [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz"
    ]
    
    for filename in files:
        filepath = data_dir / filename
        if not filepath.exists():
            url = base_url + filename
            logger.info("Downloading %s", url)
            urllib.request.urlretrieve(url, filepath)

def load_mnist(data_dir: str = 'data') -> Tuple[ArrayType, ArrayType, ArrayType, ArrayType]:
    """Load MNIST dataset.
    
    Args:
        data_dir: Directory containing dataset.
        
    Returns:
        Tuple of (train_images, train_labels, test_images, test_labels).
        
    Raises:
        FileNotFoundError: If dataset files are not found.
    """
    data_dir = Path(data_dir)
    
    def load_images(filename: str) -> ArrayType:
        filepath = data_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Dataset file not found: {filepath}")
            
        with gzip.open(filepath, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        return data.reshape(-1, 784)
    
    def load_labels(filename: str) -> ArrayType:
        filepath = data_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Dataset file not found: {filepath}")
            
        with gzip.open(filepath, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        return data
    
    # Load data
    X_train = load_images("train-images-idx3-ubyte.gz")
    y_train = load_labels("train-labels-idx1-ubyte.gz")
    X_test = load_images("t10k-images-idx3-ubyte.gz")
    y_test = load_labels("t10k-labels-idx1-ubyte.gz")
    
    logger.info("Loaded MNIST dataset: X_train=%s, y_train=%s, X_test=%s, y_test=%s",
                X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    
    return X_train, y_train, X_test, y_test

def preprocess_data(X: ArrayType, y: ArrayType) -> Tuple[ArrayType, ArrayType]:
    """Preprocess data for training.
    
    Args:
        X: Input images.
        y: Labels.
        
    Returns:
        Tuple of (preprocessed_images, one_hot_labels).
    """
    # Normalize images
    X = X.astype('float32') / 255.0
    
    # Reshape images
    if len(X.shape) == 2:
        X = X.T
    
    # Convert labels to one-hot encoding
    n_classes = 10
    y_one_hot = np.zeros((n_classes, y.shape[0]))
    y_one_hot[y, np.arange(y.shape[0])] = 1
    
    return X, y_one_hot

def plot_digits(X: ArrayType, y: Optional[ArrayType] = None,
               predictions: Optional[ArrayType] = None,
               n_samples: int = 10) -> None:
    """Plot sample digits with optional labels and predictions.
    
    Args:
        X: Input images.
        y: True labels (optional).
        predictions: Predicted labels (optional).
        n_samples: Number of samples to plot.
    """
    fig, axes = plt.subplots(1, n_samples, figsize=(2*n_samples, 2))
    
    for i in range(n_samples):
        if len(X.shape) == 2:
            img = X[i].reshape(28, 28)
        else:
            img = X[:, i].reshape(28, 28)
            
        axes[i].imshow(img, cmap='gray')
        axes[i].axis('off')
        
        title = ''
        if y is not None:
            title += f'True: {y[i]}'
        if predictions is not None:
            title += f'\nPred: {predictions[i]}'
        axes[i].set_title(title)
    
    plt.tight_layout()
    plt.show()

def plot_training_history(costs: List[float], title: str = "Training Progress") -> None:
    """Plot training cost history.
    
    Args:
        costs: List of costs per iteration.
        title: Plot title.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(costs)
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.title(title)
    plt.grid(True)
    plt.show()

def plot_confusion_matrix(y_true: ArrayType, y_pred: ArrayType) -> None:
    """Plot confusion matrix.
    
    Args:
        y_true: True labels.
        y_pred: Predicted labels.
    """
    n_classes = 10
    cm = np.zeros((n_classes, n_classes), dtype=int)
    
    for i in range(len(y_true)):
        cm[y_true[i], y_pred[i]] += 1
    
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    plt.xticks(np.arange(n_classes))
    plt.yticks(np.arange(n_classes))
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    # Add text annotations
    thresh = cm.max() / 2
    for i in range(n_classes):
        for j in range(n_classes):
            plt.text(j, i, str(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.show()

def compute_metrics(y_true: ArrayType, y_pred: ArrayType) -> Dict[str, float]:
    """Compute classification metrics.
    
    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        
    Returns:
        Dictionary containing accuracy and per-class metrics.
    """
    n_classes = 10
    metrics = {}
    
    # Overall accuracy
    metrics['accuracy'] = np.mean(y_true == y_pred)
    
    # Per-class metrics
    for i in range(n_classes):
        # True positives
        tp = np.sum((y_true == i) & (y_pred == i))
        # False positives
        fp = np.sum((y_true != i) & (y_pred == i))
        # False negatives
        fn = np.sum((y_true == i) & (y_pred != i))
        
        # Precision
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        metrics[f'precision_{i}'] = precision
        
        # Recall
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics[f'recall_{i}'] = recall
        
        # F1 score
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        metrics[f'f1_{i}'] = f1
    
    logger.info("Computed metrics: accuracy=%.3f", metrics['accuracy'])
    return metrics

def save_results(results: Dict[str, Any], filepath: str) -> None:
    """Save training/evaluation results to file.
    
    Args:
        results: Dictionary containing results.
        filepath: Path to save file.
    """
    # Convert numpy arrays to lists for JSON serialization
    serializable_results = {}
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            serializable_results[key] = value.tolist()
        elif isinstance(value, np.float32) or isinstance(value, np.float64):
            serializable_results[key] = float(value)
        else:
            serializable_results[key] = value
    
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'wb') as f:
        pickle.dump(serializable_results, f)
    
    logger.info("Saved results to %s", filepath) 