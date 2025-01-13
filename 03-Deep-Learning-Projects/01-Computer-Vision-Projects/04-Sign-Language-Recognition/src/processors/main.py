"""Main processing functions for model training and evaluation."""

import os
from typing import Dict, List, Optional, Tuple, Union
import tensorflow as tf
import numpy as np
import logging
from ..core import SignLanguageModel, TrainingCallback
from ..utils import create_dataset, load_and_preprocess_image, plot_training_history, plot_confusion_matrix

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_model(
    data_dir: str,
    num_classes: int,
    model_dir: str = 'models',
    image_size: Tuple[int, int] = (64, 64),
    batch_size: int = 32,
    epochs: int = 20,
    learning_rate: float = 0.001,
    validation_split: float = 0.2,
    seed: int = 42
) -> Dict[str, List[float]]:
    """Train sign language recognition model.
    
    Args:
        data_dir: Directory containing class subdirectories
        num_classes: Number of sign language classes
        model_dir: Directory to save model weights
        image_size: Target image size
        batch_size: Training batch size
        epochs: Number of training epochs
        learning_rate: Initial learning rate
        validation_split: Fraction of data for validation
        seed: Random seed for reproducibility
        
    Returns:
        Training history
        
    Raises:
        ValueError: If parameters or data are invalid
    """
    # Create datasets
    logger.info("Creating datasets...")
    train_ds, val_ds = create_dataset(
        data_dir,
        image_size=image_size,
        batch_size=batch_size,
        validation_split=validation_split,
        seed=seed
    )
    
    # Create and train model
    logger.info("Initializing model...")
    model = SignLanguageModel(
        num_classes=num_classes,
        input_shape=(*image_size, 1),
        learning_rate=learning_rate
    )
    
    # Train model
    logger.info("Starting model training...")
    history = model.train(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )
    
    # Save model
    os.makedirs(model_dir, exist_ok=True)
    weights_path = os.path.join(model_dir, 'model_weights.h5')
    model.save_weights(weights_path)
    
    return history

def evaluate_model(
    model: SignLanguageModel,
    data_dir: str,
    class_names: List[str],
    image_size: Tuple[int, int] = (64, 64),
    batch_size: int = 32,
    plot_results: bool = True
) -> Tuple[float, float]:
    """Evaluate model on test data.
    
    Args:
        model: Trained SignLanguageModel
        data_dir: Directory containing test data
        class_names: List of class names
        image_size: Target image size
        batch_size: Evaluation batch size
        plot_results: Whether to plot confusion matrix
        
    Returns:
        Tuple of (loss, accuracy)
        
    Raises:
        ValueError: If data directory is invalid
    """
    # Create test dataset
    logger.info("Creating test dataset...")
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        image_size=image_size,
        batch_size=batch_size,
        color_mode='grayscale',
        shuffle=False
    )
    
    # Get predictions
    y_true = []
    y_pred = []
    for images, labels in test_ds:
        images = tf.cast(images, tf.float32) / 255.0
        predictions = model.model.predict(images)
        y_true.extend(tf.argmax(labels, axis=1).numpy())
        y_pred.extend(tf.argmax(predictions, axis=1).numpy())
    
    # Evaluate
    loss, accuracy = model.evaluate(test_ds)
    
    # Plot confusion matrix if requested
    if plot_results:
        plot_confusion_matrix(
            np.array(y_true),
            np.array(y_pred),
            class_names
        )
    
    return loss, accuracy

def predict_image(
    model: SignLanguageModel,
    image_path: str,
    class_names: List[str],
    image_size: Tuple[int, int] = (64, 64)
) -> Tuple[str, float]:
    """Make prediction on a single image.
    
    Args:
        model: Trained SignLanguageModel
        image_path: Path to image file
        class_names: List of class names
        image_size: Target image size
        
    Returns:
        Tuple of (predicted_class, confidence)
        
    Raises:
        ValueError: If image file or class names are invalid
    """
    if not class_names:
        raise ValueError("class_names must not be empty")
    
    # Load and preprocess image
    image = load_and_preprocess_image(image_path, image_size)
    
    # Get prediction
    probabilities = model.predict(image, return_probabilities=True)
    predicted_idx = int(tf.argmax(probabilities))
    confidence = float(probabilities[predicted_idx])
    
    predicted_class = class_names[predicted_idx]
    logger.info(f"Predicted sign: {predicted_class} with confidence: {confidence:.4f}")
    
    return predicted_class, confidence 