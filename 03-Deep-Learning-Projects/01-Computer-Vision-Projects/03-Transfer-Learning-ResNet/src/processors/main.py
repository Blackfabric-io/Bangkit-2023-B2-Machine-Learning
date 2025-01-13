"""Main processing functions for model training and evaluation."""

import os
from typing import Dict, List, Optional, Tuple, Union
import tensorflow as tf
import logging
from ..core import ResNetTransfer, TrainingCallback
from ..utils import create_dataset, load_and_preprocess_image, plot_training_history

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_model(
    data_dir: str,
    num_classes: int,
    model_dir: str = 'models',
    image_size: Tuple[int, int] = (224, 224),
    batch_size: int = 32,
    epochs: int = 10,
    learning_rate: float = 0.001,
    validation_split: float = 0.2,
    fine_tune: bool = False,
    fine_tune_epochs: int = 5,
    seed: int = 42
) -> Dict[str, List[float]]:
    """Train ResNet transfer learning model.
    
    Args:
        data_dir: Directory containing class subdirectories
        num_classes: Number of target classes
        model_dir: Directory to save model weights
        image_size: Target image size
        batch_size: Training batch size
        epochs: Number of training epochs
        learning_rate: Initial learning rate
        validation_split: Fraction of data for validation
        fine_tune: Whether to perform fine-tuning
        fine_tune_epochs: Number of fine-tuning epochs
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
    model = ResNetTransfer(
        num_classes=num_classes,
        input_shape=(*image_size, 3),
        learning_rate=learning_rate
    )
    
    # Initial training
    logger.info("Starting initial training...")
    history = model.train(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )
    
    # Fine-tuning if requested
    if fine_tune:
        logger.info("Starting fine-tuning...")
        model.unfreeze_layers()
        fine_tune_history = model.train(
            train_ds,
            validation_data=val_ds,
            epochs=fine_tune_epochs
        )
        
        # Combine histories
        for k in history.keys():
            history[k].extend(fine_tune_history[k])
    
    # Save model
    os.makedirs(model_dir, exist_ok=True)
    weights_path = os.path.join(model_dir, 'model_weights.h5')
    model.save_weights(weights_path)
    
    return history

def evaluate_model(
    model: ResNetTransfer,
    data_dir: str,
    image_size: Tuple[int, int] = (224, 224),
    batch_size: int = 32
) -> Tuple[float, float]:
    """Evaluate model on test data.
    
    Args:
        model: Trained ResNetTransfer model
        data_dir: Directory containing test data
        image_size: Target image size
        batch_size: Evaluation batch size
        
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
        shuffle=False
    )
    
    # Evaluate
    logger.info("Evaluating model...")
    loss, accuracy = model.model.evaluate(test_ds)
    
    logger.info(f"Test Loss: {loss:.4f}")
    logger.info(f"Test Accuracy: {accuracy:.4f}")
    
    return loss, accuracy

def predict_image(
    model: ResNetTransfer,
    image_path: str,
    class_names: List[str],
    image_size: Tuple[int, int] = (224, 224)
) -> Tuple[str, float]:
    """Make prediction on a single image.
    
    Args:
        model: Trained ResNetTransfer model
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
    logger.info(f"Predicted class: {predicted_class} with confidence: {confidence:.4f}")
    
    return predicted_class, confidence 