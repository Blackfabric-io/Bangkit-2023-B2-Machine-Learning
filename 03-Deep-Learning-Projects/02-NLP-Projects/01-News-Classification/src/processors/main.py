"""Main processing functions for text classification."""

import logging
from typing import Dict, List, Tuple, Optional
import tensorflow as tf
import numpy as np
from ..core import TextPreprocessor, TokenizerWrapper
from ..utils import plot_metrics, plot_confusion_matrix

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_data(
    train_file: str,
    test_file: Optional[str] = None,
    max_sequence_length: Optional[int] = None
) -> Tuple[Dict[str, np.ndarray], TokenizerWrapper]:
    """Process text data for training and testing.
    
    Args:
        train_file: Path to training data file
        test_file: Optional path to test data file
        max_sequence_length: Maximum sequence length for padding
        
    Returns:
        Tuple of (data_dict, tokenizer)
        
    Raises:
        FileNotFoundError: If data files don't exist
    """
    # Initialize processors
    text_processor = TextPreprocessor()
    tokenizer = TokenizerWrapper()
    
    # Process training data
    logger.info("Processing training data...")
    train_texts, train_labels = text_processor.process_text_file(train_file)
    
    # Fit tokenizers
    tokenizer.fit_text_tokenizer(train_texts)
    tokenizer.fit_label_tokenizer(train_labels)
    
    # Convert to sequences
    train_sequences = tokenizer.get_padded_sequences(
        train_texts,
        maxlen=max_sequence_length
    )
    train_label_sequences = tokenizer.get_label_sequences(train_labels)
    
    # Prepare output dictionary
    data = {
        'train_sequences': train_sequences,
        'train_labels': train_label_sequences
    }
    
    # Process test data if provided
    if test_file:
        logger.info("Processing test data...")
        test_texts, test_labels = text_processor.process_text_file(test_file)
        
        test_sequences = tokenizer.get_padded_sequences(
            test_texts,
            maxlen=max_sequence_length
        )
        test_label_sequences = tokenizer.get_label_sequences(test_labels)
        
        data.update({
            'test_sequences': test_sequences,
            'test_labels': test_label_sequences
        })
    
    return data, tokenizer

def train_model(
    train_sequences: np.ndarray,
    train_labels: List[int],
    num_classes: int,
    embedding_dim: int = 100,
    epochs: int = 10,
    batch_size: int = 32,
    validation_split: float = 0.2
) -> Tuple[tf.keras.Model, Dict[str, List[float]]]:
    """Train text classification model.
    
    Args:
        train_sequences: Training sequences
        train_labels: Training labels
        num_classes: Number of classes
        embedding_dim: Embedding dimension
        epochs: Number of training epochs
        batch_size: Training batch size
        validation_split: Validation data fraction
        
    Returns:
        Tuple of (trained_model, training_history)
    """
    # Convert labels to categorical
    train_labels = tf.keras.utils.to_categorical(train_labels)
    
    # Build model
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(
            input_dim=len(tokenizer.word_index) + 1,
            output_dim=embedding_dim,
            input_length=train_sequences.shape[1]
        ),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(24, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compile model
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    
    # Train model
    logger.info("Starting model training...")
    history = model.fit(
        train_sequences,
        train_labels,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split
    )
    
    # Plot training metrics
    plot_metrics(history.history)
    
    return model, history.history

def evaluate_model(
    model: tf.keras.Model,
    test_sequences: np.ndarray,
    test_labels: List[int],
    class_names: List[str]
) -> Tuple[float, float]:
    """Evaluate model on test data.
    
    Args:
        model: Trained model
        test_sequences: Test sequences
        test_labels: Test labels
        class_names: Names of classes
        
    Returns:
        Tuple of (loss, accuracy)
    """
    # Convert labels to categorical
    test_labels_cat = tf.keras.utils.to_categorical(test_labels)
    
    # Evaluate model
    logger.info("Evaluating model...")
    loss, accuracy = model.evaluate(test_sequences, test_labels_cat)
    
    # Get predictions
    predictions = model.predict(test_sequences)
    predicted_labels = np.argmax(predictions, axis=1)
    
    # Plot confusion matrix
    plot_confusion_matrix(test_labels, predicted_labels, class_names)
    
    logger.info(f"Test Loss: {loss:.4f}")
    logger.info(f"Test Accuracy: {accuracy:.4f}")
    
    return loss, accuracy 