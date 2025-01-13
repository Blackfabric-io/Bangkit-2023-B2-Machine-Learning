"""
Visualization utilities for TensorBoard and confusion matrix plotting.
"""

from typing import List, Optional
import io
import logging
import itertools
from datetime import datetime
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import sklearn.metrics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TensorBoardLogger:
    """
    TensorBoard logging utilities for model visualization.
    """
    
    def __init__(
        self,
        log_dir: str = "logs/fashion_mnist",
        class_names: Optional[List[str]] = None
    ) -> None:
        """
        Initialize TensorBoard logger.
        
        Args:
            log_dir: Directory to store TensorBoard logs
            class_names: List of class names for confusion matrix
        """
        try:
            # Create log directories
            self.log_dir = f"{log_dir}/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            self.cm_log_dir = f"{self.log_dir}/cm"
            
            # Create file writers
            self.file_writer_cm = tf.summary.create_file_writer(self.cm_log_dir)
            
            # Store class names
            self.class_names = class_names
            
            # Create callbacks
            self.tensorboard_callback = keras.callbacks.TensorBoard(
                log_dir=self.log_dir
            )
            self.confusion_matrix_callback = keras.callbacks.LambdaCallback(
                on_epoch_end=self._log_confusion_matrix
            )
            
            logger.info(f"TensorBoard logger initialized at {self.log_dir}")
            
        except Exception as e:
            logger.error(f"Error initializing TensorBoard logger: {str(e)}")
            raise
            
    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        class_names: List[str]
    ) -> plt.Figure:
        """
        Create a matplotlib figure containing the plotted confusion matrix.
        
        Args:
            cm: Confusion matrix array
            class_names: List of class names
            
        Returns:
            Matplotlib figure object
        """
        try:
            figure = plt.figure(figsize=(8, 8))
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title("Confusion matrix")
            plt.colorbar()
            
            # Set up axes
            tick_marks = np.arange(len(class_names))
            plt.xticks(tick_marks, class_names, rotation=45)
            plt.yticks(tick_marks, class_names)
            
            # Normalize the confusion matrix
            cm = np.around(
                cm.astype('float') / cm.sum(axis=1)[:, np.newaxis],
                decimals=2
            )
            
            # Add text annotations
            threshold = cm.max() / 2.
            for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                color = "white" if cm[i, j] > threshold else "black"
                plt.text(
                    j, i, cm[i, j],
                    horizontalalignment="center",
                    color=color
                )
                
            plt.tight_layout()
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            
            return figure
            
        except Exception as e:
            logger.error(f"Error plotting confusion matrix: {str(e)}")
            raise
            
    def plot_to_image(self, figure: plt.Figure) -> tf.Tensor:
        """
        Convert a Matplotlib figure to a TensorFlow image tensor.
        
        Args:
            figure: Matplotlib figure object
            
        Returns:
            TensorFlow image tensor
        """
        try:
            # Save the plot to a PNG in memory
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            
            # Close the figure to free memory
            plt.close(figure)
            buf.seek(0)
            
            # Convert PNG buffer to TF image
            image = tf.image.decode_png(buf.getvalue(), channels=4)
            
            # Add batch dimension
            image = tf.expand_dims(image, 0)
            
            return image
            
        except Exception as e:
            logger.error(f"Error converting plot to image: {str(e)}")
            raise
            
    def _log_confusion_matrix(
        self,
        epoch: int,
        logs: dict
    ) -> None:
        """
        Log confusion matrix at the end of each epoch.
        
        Args:
            epoch: Current epoch number
            logs: Dictionary containing the training metrics
        """
        try:
            # Get the current model from the training context
            model = keras.backend.get_value(self.model)
            
            # Get predictions
            test_pred = model.predict(self.test_images)
            test_pred = np.argmax(test_pred, axis=1)
            
            # Calculate confusion matrix
            cm = sklearn.metrics.confusion_matrix(
                self.test_labels,
                test_pred
            )
            
            # Create figure
            figure = self.plot_confusion_matrix(cm, self.class_names)
            cm_image = self.plot_to_image(figure)
            
            # Log to TensorBoard
            with self.file_writer_cm.as_default():
                tf.summary.image(
                    "Confusion Matrix",
                    cm_image,
                    step=epoch
                )
                
        except Exception as e:
            logger.error(f"Error logging confusion matrix: {str(e)}")
            raise 