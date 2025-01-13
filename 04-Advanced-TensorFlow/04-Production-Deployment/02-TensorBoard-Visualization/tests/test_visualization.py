"""
Tests for visualization utilities.
"""

import os
import tempfile
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from src.utils.visualization import TensorBoardLogger

def test_tensorboard_logger_initialization():
    """Test TensorBoard logger initialization."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        logger = TensorBoardLogger(
            log_dir=tmp_dir,
            class_names=['class1', 'class2']
        )
        
        # Check if log directories are created
        assert os.path.exists(logger.log_dir)
        assert os.path.exists(logger.cm_log_dir)
        
        # Check if callbacks are created
        assert isinstance(logger.tensorboard_callback,
                        tf.keras.callbacks.TensorBoard)
        assert isinstance(logger.confusion_matrix_callback,
                        tf.keras.callbacks.LambdaCallback)

def test_plot_confusion_matrix():
    """Test confusion matrix plotting."""
    # Create sample confusion matrix
    cm = np.array([[5, 1], [2, 8]])
    class_names = ['class1', 'class2']
    
    logger = TensorBoardLogger(class_names=class_names)
    figure = logger.plot_confusion_matrix(cm, class_names)
    
    # Check if figure is created
    assert isinstance(figure, plt.Figure)
    
    # Clean up
    plt.close(figure)

def test_plot_to_image():
    """Test conversion of plot to image tensor."""
    # Create sample figure
    figure = plt.figure(figsize=(8, 8))
    plt.plot([1, 2, 3], [1, 2, 3])
    
    logger = TensorBoardLogger()
    image = logger.plot_to_image(figure)
    
    # Check image tensor properties
    assert isinstance(image, tf.Tensor)
    assert len(image.shape) == 4  # [batch, height, width, channels]
    assert image.shape[-1] == 4  # RGBA channels

def test_log_confusion_matrix():
    """Test confusion matrix logging."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        logger = TensorBoardLogger(
            log_dir=tmp_dir,
            class_names=['class1', 'class2']
        )
        
        # Create dummy test data
        logger.test_images = np.random.rand(10, 28, 28, 1)
        logger.test_labels = np.random.randint(0, 2, size=10)
        
        # Create dummy model
        model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
            tf.keras.layers.Dense(2, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        
        # Set model in logger
        logger.model = model
        
        # Test logging
        logger._log_confusion_matrix(epoch=0, logs={})
        
        # Check if log file is created
        assert len(os.listdir(logger.cm_log_dir)) > 0 