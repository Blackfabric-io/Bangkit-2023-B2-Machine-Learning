"""Core ResNet transfer learning implementation."""

from typing import Tuple, Optional, Dict, Any, List, Union
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import Callback
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrainingCallback(Callback):
    """Custom callback for monitoring training progress."""
    
    def __init__(self, 
                 accuracy_threshold: float = 0.95,
                 patience: int = 3) -> None:
        """Initialize callback.
        
        Args:
            accuracy_threshold: Training accuracy threshold to stop at
            patience: Number of epochs to wait for improvement
        """
        super().__init__()
        self.accuracy_threshold = accuracy_threshold
        self.patience = patience
        self.wait = 0
        self.best_acc = 0
        
    def on_epoch_end(self, epoch: int, logs: Dict[str, float] = None) -> None:
        """Check metrics at epoch end.
        
        Args:
            epoch: Current epoch number
            logs: Training metrics
        """
        current_acc = logs.get('accuracy', 0)
        
        # Check if accuracy threshold reached
        if current_acc >= self.accuracy_threshold:
            logger.info(f'Accuracy threshold {self.accuracy_threshold} reached. Stopping training.')
            self.model.stop_training = True
            return
            
        # Check for improvement
        if current_acc > self.best_acc:
            self.best_acc = current_acc
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                logger.info(f'No improvement for {self.patience} epochs. Stopping training.')
                self.model.stop_training = True

class ResNetTransfer:
    """ResNet50 transfer learning model for image classification."""
    
    def __init__(self,
                 num_classes: int,
                 input_shape: Tuple[int, int, int] = (224, 224, 3),
                 weights: str = 'imagenet',
                 learning_rate: float = 0.001) -> None:
        """Initialize the model.
        
        Args:
            num_classes: Number of target classes
            input_shape: Input image shape (height, width, channels)
            weights: Pre-trained weights to use
            learning_rate: Learning rate for training
            
        Raises:
            ValueError: If parameters are invalid
        """
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")
            
        if any(dim <= 0 for dim in input_shape):
            raise ValueError("input_shape dimensions must be positive")
            
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.weights = weights
        self.learning_rate = learning_rate
        self.model = self._build_model()
        
    def _build_model(self) -> Model:
        """Build and compile the model.
        
        Returns:
            Compiled Keras model
        """
        # Load pre-trained ResNet
        base_model = ResNet50(
            weights=self.weights,
            include_top=False,
            input_shape=self.input_shape
        )
        
        # Freeze base model layers
        for layer in base_model.layers:
            layer.trainable = False
            
        # Add custom top layers
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        predictions = Dense(self.num_classes, activation='softmax')(x)
        
        # Create model
        model = Model(inputs=base_model.input, outputs=predictions)
        
        # Compile
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self,
              train_data: tf.data.Dataset,
              validation_data: Optional[tf.data.Dataset] = None,
              epochs: int = 10,
              callbacks: Optional[List[Callback]] = None) -> Dict[str, List[float]]:
        """Train the model.
        
        Args:
            train_data: Training dataset
            validation_data: Optional validation dataset
            epochs: Number of epochs to train
            callbacks: Optional list of callbacks
            
        Returns:
            Training history
            
        Raises:
            ValueError: If training data format is invalid
        """
        if not isinstance(train_data, tf.data.Dataset):
            raise ValueError("train_data must be a tf.data.Dataset")
            
        if validation_data is not None and not isinstance(validation_data, tf.data.Dataset):
            raise ValueError("validation_data must be a tf.data.Dataset")
        
        # Add default callback if none provided
        if callbacks is None:
            callbacks = [TrainingCallback()]
            
        # Train model
        logger.info("Starting model training...")
        history = self.model.fit(
            train_data,
            validation_data=validation_data,
            epochs=epochs,
            callbacks=callbacks
        )
        
        return history.history
    
    def predict(self, 
                image: Union[tf.Tensor, np.ndarray],
                return_probabilities: bool = False) -> Union[int, np.ndarray]:
        """Make prediction on a single image.
        
        Args:
            image: Input image
            return_probabilities: Whether to return class probabilities
            
        Returns:
            Predicted class index or probability distribution
            
        Raises:
            ValueError: If image format is invalid
        """
        if not isinstance(image, (tf.Tensor, np.ndarray)):
            raise ValueError("Input must be a tensor or numpy array")
            
        # Ensure batch dimension
        if len(image.shape) == 3:
            image = tf.expand_dims(image, 0)
            
        # Make prediction
        predictions = self.model.predict(image)
        
        if return_probabilities:
            return predictions[0]
        else:
            return int(tf.argmax(predictions[0]))
    
    def save_weights(self, filepath: str) -> None:
        """Save model weights.
        
        Args:
            filepath: Path to save weights to
        """
        self.model.save_weights(filepath)
        logger.info(f"Model weights saved to {filepath}")
    
    def load_weights(self, filepath: str) -> None:
        """Load model weights.
        
        Args:
            filepath: Path to load weights from
            
        Raises:
            ValueError: If weights file doesn't exist
        """
        try:
            self.model.load_weights(filepath)
            logger.info(f"Model weights loaded from {filepath}")
        except Exception as e:
            raise ValueError(f"Failed to load weights: {str(e)}")
    
    def unfreeze_layers(self, from_layer: int = -10) -> None:
        """Unfreeze layers for fine-tuning.
        
        Args:
            from_layer: Index to start unfreezing from (negative counts from end)
        """
        base_model = self.model.layers[0]  # ResNet base
        
        # Unfreeze specified layers
        for layer in base_model.layers[from_layer:]:
            layer.trainable = True
            
        # Recompile with lower learning rate
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate/10),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info(f"Unfroze layers from index {from_layer}. Model recompiled with lower learning rate.") 