"""Helper functions for data loading and visualization."""

import os
from typing import Tuple, List, Dict, Optional
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from PIL import Image

def create_dataset(
    data_dir: str,
    image_size: Tuple[int, int] = (64, 64),
    batch_size: int = 32,
    validation_split: float = 0.2,
    seed: int = 42
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """Create training and validation datasets.
    
    Args:
        data_dir: Directory containing class subdirectories
        image_size: Target image size
        batch_size: Batch size for training
        validation_split: Fraction of data to use for validation
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_dataset, validation_dataset)
        
    Raises:
        ValueError: If directory structure is invalid
    """
    if not os.path.exists(data_dir):
        raise ValueError(f"Data directory not found: {data_dir}")
    
    # Create dataset from directory
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
        subset="both",
        seed=seed,
        image_size=image_size,
        batch_size=batch_size,
        color_mode='grayscale'
    )
    
    train_ds, val_ds = dataset
    
    # Normalize pixel values
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
    
    # Configure for performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    
    return train_ds, val_ds

def load_and_preprocess_image(
    image_path: str,
    target_size: Tuple[int, int] = (64, 64)
) -> tf.Tensor:
    """Load and preprocess a single image.
    
    Args:
        image_path: Path to image file
        target_size: Size to resize image to
        
    Returns:
        Preprocessed image tensor
        
    Raises:
        ValueError: If image file doesn't exist or can't be processed
    """
    if not os.path.exists(image_path):
        raise ValueError(f"Image file not found: {image_path}")
    
    try:
        # Read image
        img = tf.io.read_file(image_path)
        img = tf.image.decode_image(img, channels=1)
        
        # Resize
        img = tf.image.resize(img, target_size)
        
        # Convert to float and normalize
        img = tf.cast(img, tf.float32) / 255.0
        
        # Add batch dimension
        img = tf.expand_dims(img, 0)
        
        return img
        
    except Exception as e:
        raise ValueError(f"Failed to process image: {str(e)}")

def plot_training_history(
    history: Dict[str, List[float]],
    figsize: Tuple[int, int] = (12, 4)
) -> None:
    """Plot training history metrics.
    
    Args:
        history: Training history dictionary
        figsize: Figure size for the plot
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

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    figsize: Tuple[int, int] = (10, 8)
) -> None:
    """Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        figsize: Figure size for the plot
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot
    plt.figure(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.show() 