from typing import Dict, List, Optional, Tuple, Union
import tensorflow as tf
import pandas as pd
from tensorflow.keras import layers, Model
from tensorflow import feature_column

def df_to_dataset(dataframe: pd.DataFrame,
                 shuffle: bool = True,
                 batch_size: int = 32) -> tf.data.Dataset:
    """Convert a Pandas DataFrame to a TensorFlow Dataset.
    
    Args:
        dataframe: Input pandas DataFrame
        shuffle: Whether to shuffle the data
        batch_size: Size of batches to create
        
    Returns:
        A tf.data.Dataset
    """
    dataframe = dataframe.copy()
    labels = dataframe.pop('target')
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels.values))
    
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    
    return ds

class StructuredDataModel:
    """A model for structured data classification using feature columns."""
    
    def __init__(self,
                 numeric_features: List[str],
                 categorical_features: Dict[str, List[str]],
                 embedding_dims: Optional[Dict[str, int]] = None):
        """Initialize the model.
        
        Args:
            numeric_features: List of numeric column names
            categorical_features: Dict of categorical column names and their vocabularies
            embedding_dims: Optional dict of categorical columns and their embedding dimensions
        """
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.embedding_dims = embedding_dims or {}
        self.feature_columns = []
        self.model = None
        self._create_feature_columns()
    
    def _create_feature_columns(self):
        """Create feature columns for the model."""
        # Numeric columns
        for header in self.numeric_features:
            numeric_col = feature_column.numeric_column(header)
            self.feature_columns.append(numeric_col)
        
        # Categorical columns
        for col_name, vocabulary in self.categorical_features.items():
            cat_col = feature_column.categorical_column_with_vocabulary_list(
                col_name, vocabulary)
            
            if col_name in self.embedding_dims:
                # Create embedding column
                embed_col = feature_column.embedding_column(
                    cat_col, dimension=self.embedding_dims[col_name])
                self.feature_columns.append(embed_col)
            else:
                # Create one-hot encoded column
                indicator_col = feature_column.indicator_column(cat_col)
                self.feature_columns.append(indicator_col)
    
    def build_model(self, hidden_units: List[int] = [128, 128]):
        """Build the Keras model.
        
        Args:
            hidden_units: List of integers, the layer sizes of the DNN
        """
        feature_layer = layers.DenseFeatures(self.feature_columns)
        
        model = tf.keras.Sequential()
        model.add(feature_layer)
        
        for units in hidden_units:
            model.add(layers.Dense(units, activation='relu'))
        
        model.add(layers.Dense(1, activation='sigmoid'))
        
        self.model = model
        return model
    
    def compile_model(self,
                     optimizer: str = 'adam',
                     loss: str = 'binary_crossentropy',
                     metrics: List[str] = ['accuracy']):
        """Compile the Keras model.
        
        Args:
            optimizer: Name of the optimizer or optimizer instance
            loss: Name of the loss function
            metrics: List of metrics to track
        """
        if self.model is None:
            raise ValueError("Model must be built before compiling")
        
        self.model.compile(optimizer=optimizer,
                          loss=loss,
                          metrics=metrics)
    
    def train(self,
              train_ds: tf.data.Dataset,
              validation_ds: Optional[tf.data.Dataset] = None,
              epochs: int = 100,
              **kwargs) -> tf.keras.callbacks.History:
        """Train the model.
        
        Args:
            train_ds: Training dataset
            validation_ds: Optional validation dataset
            epochs: Number of epochs to train
            **kwargs: Additional arguments to pass to model.fit()
            
        Returns:
            Training history
        """
        if self.model is None:
            raise ValueError("Model must be built and compiled before training")
        
        return self.model.fit(train_ds,
                            validation_data=validation_ds,
                            epochs=epochs,
                            **kwargs)
    
    def evaluate(self, test_ds: tf.data.Dataset) -> Tuple[float, float]:
        """Evaluate the model.
        
        Args:
            test_ds: Test dataset
            
        Returns:
            Tuple of (loss, accuracy)
        """
        if self.model is None:
            raise ValueError("Model must be built before evaluation")
        
        return self.model.evaluate(test_ds) 