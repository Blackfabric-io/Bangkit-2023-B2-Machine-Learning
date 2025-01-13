import tensorflow as tf
import tensorflow_datasets as tfds
from typing import Tuple, Any
import logging

logger = logging.getLogger(__name__)

def format_example(feature: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """Format and preprocess a single example.
    
    Args:
        feature: Input image tensor
        label: Input label tensor
        
    Returns:
        Tuple of (feature, one_hot_label)
    """
    # One-hot encode the label
    one_hot = tf.one_hot(label, depth=3)
    return feature, one_hot

class RockPaperScissorsModel:
    """Main class for Rock Paper Scissors model training."""
    
    def __init__(self, batch_size: int = 10):
        """Initialize the model.
        
        Args:
            batch_size: Batch size for training
        """
        self.batch_size = batch_size
        self.model = None
        self.class_names = ['Rock', 'Paper', 'Scissors']
        
    def load_data(self, data_dir: str) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        """Load and preprocess the Rock Paper Scissors dataset.
        
        Args:
            data_dir: Directory to store the dataset
            
        Returns:
            Tuple of (train_data, val_data)
        """
        logger.info("Loading Rock Paper Scissors dataset...")
        train_data = tfds.load('rock_paper_scissors', 
                             data_dir=data_dir,
                             split='train',
                             as_supervised=True)
        val_data = tfds.load('rock_paper_scissors',
                           data_dir=data_dir,
                           split='test',
                           as_supervised=True)
        
        # Apply one-hot encoding
        train_data = train_data.map(format_example)
        val_data = val_data.map(format_example)
        
        # Create batched datasets
        train_batches = train_data.shuffle(100).batch(self.batch_size)
        validation_batches = val_data.batch(32)
        
        return train_batches, validation_batches
    
    def build_model(self) -> None:
        """Build the CNN model architecture."""
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(16, (3, 3), activation='relu', 
                                 input_shape=(300, 300, 3)),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(3, activation='softmax')
        ])
        
        self.model.compile(optimizer='adam',
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
        
    def train(self, train_batches: tf.data.Dataset, 
              validation_batches: tf.data.Dataset,
              epochs: int = 3) -> tf.keras.callbacks.History:
        """Train the model.
        
        Args:
            train_batches: Training dataset
            validation_batches: Validation dataset
            epochs: Number of training epochs
            
        Returns:
            Training history
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
            
        logger.info("Starting model training...")
        return self.model.fit(train_batches,
                            epochs=epochs,
                            validation_data=validation_batches,
                            validation_steps=1)
    
    def get_model_info(self) -> None:
        """Print model summary."""
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
            
        self.model.summary() 