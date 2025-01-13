import tensorflow as tf
import tensorflow_datasets as tfds
from typing import Tuple, Any
import logging

logger = logging.getLogger(__name__)

def format_example(image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """Formats and normalizes the image data.
    
    Args:
        image: Input image tensor
        label: Input label tensor
        
    Returns:
        Tuple of normalized image and label tensors
    """
    image = tf.cast(image, tf.float32)
    image = image * 1.0/255.0
    return image, label

class FashionMNISTModel:
    """Main class for Fashion MNIST model training and conversion."""
    
    def __init__(self, batch_size: int = 256):
        """Initialize the model.
        
        Args:
            batch_size: Batch size for training
        """
        self.batch_size = batch_size
        self.model = None
        self.class_names = ['T-shirt_top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        
    def load_data(self, data_dir: str) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """Load and preprocess the Fashion MNIST dataset.
        
        Args:
            data_dir: Directory to store the dataset
            
        Returns:
            Tuple of train, validation and test datasets
        """
        logger.info("Loading Fashion MNIST dataset...")
        (train_examples, validation_examples, test_examples), info = tfds.load(
            'fashion_mnist',
            data_dir=data_dir,
            with_info=True,
            as_supervised=True,
            split=['train[:80%]', 'train[80%:90%]', 'train[90%:]']
        )
        
        # Create batched datasets
        train_batches = train_examples.cache().shuffle(info.splits['train'].num_examples//4)\
            .batch(self.batch_size).map(format_example).prefetch(1)
        validation_batches = validation_examples.cache().batch(self.batch_size).map(format_example)
        test_batches = test_examples.map(format_example).batch(1)
        
        return train_batches, validation_batches, test_batches
    
    def build_model(self) -> None:
        """Build the CNN model architecture."""
        self.model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(28,28,1)),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        
        self.model.compile(optimizer='adam',
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])
        
    def train(self, train_batches: tf.data.Dataset, 
              validation_batches: tf.data.Dataset,
              epochs: int = 10) -> tf.keras.callbacks.History:
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
    
    def convert_to_tflite(self, export_dir: str, optimization: str = 'LATENCY') -> bytes:
        """Convert the trained model to TFLite format.
        
        Args:
            export_dir: Directory to save the SavedModel
            optimization: Optimization strategy ('LATENCY', 'SIZE' or None)
            
        Returns:
            TFLite model as bytes
        """
        if self.model is None:
            raise ValueError("No model to convert. Train the model first.")
            
        # Save model in SavedModel format
        logger.info("Saving model in SavedModel format...")
        tf.saved_model.save(self.model, export_dir)
        
        # Convert to TFLite
        logger.info("Converting model to TFLite format...")
        converter = tf.lite.TFLiteConverter.from_saved_model(export_dir)
        
        # Set optimization
        if optimization == 'LATENCY':
            converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_LATENCY]
        elif optimization == 'SIZE':
            converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
            
        return converter.convert() 