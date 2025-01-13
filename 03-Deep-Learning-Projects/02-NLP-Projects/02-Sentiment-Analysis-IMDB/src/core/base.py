"""Core classes and functions for sentiment analysis."""

from typing import List, Tuple
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

class SentimentAnalyzer:
    """Main class for sentiment analysis."""
    
    def __init__(self, 
                 num_words: int = 1000,
                 embedding_dim: int = 16,
                 maxlen: int = 120,
                 padding: str = 'post',
                 oov_token: str = "<OOV>"):
        """Initialize the sentiment analyzer.
        
        Args:
            num_words: Maximum number of words to keep
            embedding_dim: Dimensionality of the dense embedding
            maxlen: Maximum length of all sequences
            padding: Padding strategy ('pre' or 'post')
            oov_token: Token for out-of-vocabulary words
        """
        self.num_words = num_words
        self.embedding_dim = embedding_dim
        self.maxlen = maxlen
        self.padding = padding
        self.oov_token = oov_token
        self.tokenizer = Tokenizer(num_words=num_words, oov_token=oov_token)
        self.model = None
        
    def create_model(self, num_classes: int) -> tf.keras.Model:
        """Create and compile the model.
        
        Args:
            num_classes: Number of output classes
            
        Returns:
            Compiled tensorflow model
        """
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(self.num_words, self.embedding_dim),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(24, activation='relu'),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(loss='sparse_categorical_crossentropy',
                     optimizer='adam',
                     metrics=['accuracy'])
        
        self.model = model
        return model
    
    def prepare_data(self, sentences: List[str], labels: List[str] = None) -> Tuple:
        """Prepare data for training or prediction.
        
        Args:
            sentences: List of input texts
            labels: Optional list of labels
            
        Returns:
            Tuple of (padded_sequences, labels)
        """
        self.tokenizer.fit_on_texts(sentences)
        sequences = self.tokenizer.texts_to_sequences(sentences)
        padded = pad_sequences(sequences, 
                             maxlen=self.maxlen,
                             padding=self.padding)
        
        if labels is not None:
            return padded, np.array(labels)
        return padded
    
    def train(self, 
             X_train: np.ndarray,
             y_train: np.ndarray,
             validation_data: Tuple = None,
             epochs: int = 10,
             batch_size: int = 32) -> tf.keras.callbacks.History:
        """Train the model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            validation_data: Optional tuple of validation data
            epochs: Number of epochs
            batch_size: Batch size
            
        Returns:
            Training history
        """
        if self.model is None:
            raise ValueError("Model not created. Call create_model first.")
            
        return self.model.fit(X_train,
                            y_train,
                            epochs=epochs,
                            batch_size=batch_size,
                            validation_data=validation_data)
    
    def predict(self, texts: List[str]) -> np.ndarray:
        """Make predictions on new texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            Array of predictions
        """
        if self.model is None:
            raise ValueError("Model not trained. Train the model first.")
            
        padded = self.prepare_data(texts)
        return self.model.predict(padded) 