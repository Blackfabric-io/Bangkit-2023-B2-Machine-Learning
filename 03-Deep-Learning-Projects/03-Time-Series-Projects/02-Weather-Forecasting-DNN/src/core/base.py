"""Core DNN model implementation for weather forecasting."""

from typing import Tuple, List, Optional, Union
import numpy as np
import numpy.typing as npt
import tensorflow as tf
from tensorflow import keras

class WeatherModel:
    """Deep Neural Network model for weather forecasting."""
    
    def __init__(self, 
                 input_width: int = 5,
                 label_width: int = 1,
                 shift: int = 1,
                 batch_size: int = 32) -> None:
        """Initialize the weather forecasting model.
        
        Args:
            input_width: Number of input time steps
            label_width: Number of prediction time steps
            shift: Offset between input and prediction
            batch_size: Training batch size
        """
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift
        self.batch_size = batch_size
        self.total_window_size = input_width + shift
        
        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]
        
        self.model = None
        
    def split_window(self, 
                    features: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Split windows of time into inputs and labels.
        
        Args:
            features: Input features tensor
            
        Returns:
            Tuple containing:
                - Input windows
                - Label windows
        """
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]

        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels
    
    def make_dataset(self, 
                    data: npt.NDArray[np.float32],
                    shuffle: bool = True) -> tf.data.Dataset:
        """Create TensorFlow dataset from numpy array.
        
        Args:
            data: Input data array
            shuffle: Whether to shuffle the data
            
        Returns:
            TensorFlow dataset
        """
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=shuffle,
            batch_size=self.batch_size)
        
        ds = ds.map(self.split_window)
        return ds
    
    def compile_and_fit(self,
                       train_data: tf.data.Dataset,
                       val_data: tf.data.Dataset,
                       num_features: int,
                       patience: int = 10,
                       max_epochs: int = 100) -> tf.keras.callbacks.History:
        """Compile and train the model.
        
        Args:
            train_data: Training dataset
            val_data: Validation dataset
            num_features: Number of input features
            patience: Early stopping patience
            max_epochs: Maximum training epochs
            
        Returns:
            Training history
        """
        if self.model is None:
            self.build_model(num_features)
            
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            mode='min')

        history = self.model.fit(
            train_data,
            epochs=max_epochs,
            validation_data=val_data,
            callbacks=[early_stopping])
            
        return history
    
    def build_model(self, num_features: int) -> None:
        """Build the DNN model architecture.
        
        Args:
            num_features: Number of input features
        """
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(num_features)
        ])
        
        self.model.compile(loss=tf.keras.losses.MeanSquaredError(),
                         optimizer=tf.keras.optimizers.Adam(),
                         metrics=[tf.keras.metrics.MeanAbsoluteError()])
    
    def predict(self, inputs: Union[tf.data.Dataset, npt.NDArray[np.float32]]) -> npt.NDArray[np.float32]:
        """Make predictions using the trained model.
        
        Args:
            inputs: Input data
            
        Returns:
            Model predictions
            
        Raises:
            ValueError: If model hasn't been trained
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
            
        if isinstance(inputs, np.ndarray):
            inputs = self.make_dataset(inputs)
            
        return self.model.predict(inputs) 