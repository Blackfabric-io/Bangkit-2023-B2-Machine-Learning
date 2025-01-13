import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
from typing import Tuple, List, Any
import logging

logger = logging.getLogger(__name__)

def format_image(features: dict) -> Tuple[tf.Tensor, tf.Tensor]:
    """Format and preprocess a single image.
    
    Args:
        features: Dictionary containing image and label
        
    Returns:
        Tuple of (normalized_image, label)
    """
    image = features['image']
    image = tf.image.resize(image, (224, 224)) / 255.0
    return image, features['label']

class CatDogClassifier:
    """Main class for Cats vs Dogs classifier using transfer learning."""
    
    def __init__(self, batch_size: int = 16, num_examples: int = 500):
        """Initialize the model.
        
        Args:
            batch_size: Batch size for training
            num_examples: Number of examples to use for training
        """
        self.batch_size = batch_size
        self.num_examples = num_examples
        self.model = None
        self.feature_extractor = None
        
    def load_feature_extractor(self, model_path: str) -> None:
        """Load MobileNet feature extractor.
        
        Args:
            model_path: Path to the saved MobileNet model
        """
        logger.info("Loading MobileNet feature extractor...")
        self.feature_extractor = hub.KerasLayer(model_path,
                                              input_shape=(224, 224, 3))
        self.feature_extractor.trainable = True
        
    def load_data(self, data_dir: str) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """Load and preprocess the Cats vs Dogs dataset.
        
        Args:
            data_dir: Directory to store the dataset
            
        Returns:
            Tuple of (train_batches, validation_batches, test_batches)
        """
        logger.info("Loading Cats vs Dogs dataset...")
        splits = ['train[:10%]', 'train[90%:95%]', 'train[95%:]']
        
        splits_data, info = tfds.load('cats_vs_dogs',
                                    data_dir=data_dir,
                                    split=splits,
                                    with_info=True)
        
        train_examples, validation_examples, test_examples = splits_data
        
        # Create batched datasets
        train_batches = train_examples.shuffle(self.num_examples)\
            .map(format_image).batch(self.batch_size)
        validation_batches = validation_examples.map(format_image).batch(self.batch_size)
        test_batches = test_examples.map(format_image).batch(self.batch_size)
        
        return train_batches, validation_batches, test_batches
    
    def build_model(self) -> None:
        """Build the transfer learning model architecture."""
        if self.feature_extractor is None:
            raise ValueError("Feature extractor not loaded. Call load_feature_extractor() first.")
            
        self.model = tf.keras.Sequential([
            self.feature_extractor,
            tf.keras.layers.Dense(2, activation='softmax')
        ])
        
        self.model.compile(optimizer='adam',
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])
        
    def train(self, train_batches: tf.data.Dataset,
              validation_batches: tf.data.Dataset,
              epochs: int = 5) -> tf.keras.callbacks.History:
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
                            validation_data=validation_batches)
    
    def evaluate(self, test_batches: tf.data.Dataset) -> List[float]:
        """Evaluate the model.
        
        Args:
            test_batches: Test dataset
            
        Returns:
            List of metrics
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
            
        logger.info("Evaluating model...")
        return self.model.evaluate(test_batches)
    
    def get_model_info(self) -> None:
        """Print model summary."""
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
            
        self.model.summary() 