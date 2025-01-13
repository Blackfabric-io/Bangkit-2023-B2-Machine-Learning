"""Core classes and functions for text generation."""

from typing import List, Optional, Dict, Any
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

class TextGenerator:
    """Main class for Shakespeare text generation."""
    
    def __init__(self, 
                 vocab_size: int = 1000,
                 embedding_dim: int = 100,
                 rnn_units: int = 256,
                 maxlen: int = 40,
                 padding: str = 'post',
                 oov_token: str = "<OOV>"):
        """Initialize the text generator.
        
        Args:
            vocab_size: Maximum number of words in vocabulary
            embedding_dim: Dimensionality of the dense embedding
            rnn_units: Number of RNN units
            maxlen: Maximum length of sequences
            padding: Padding strategy ('pre' or 'post')
            oov_token: Token for out-of-vocabulary words
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.rnn_units = rnn_units
        self.maxlen = maxlen
        self.padding = padding
        self.oov_token = oov_token
        self.tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
        self.model = None
        
    def create_model(self, temperature: float = 1.0) -> tf.keras.Model:
        """Create and compile the model.
        
        Args:
            temperature: Sampling temperature for generation
            
        Returns:
            Compiled tensorflow model
        """
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(self.vocab_size, self.embedding_dim),
            tf.keras.layers.LSTM(self.rnn_units, return_sequences=True),
            tf.keras.layers.Dense(self.vocab_size, activation='softmax')
        ])
        
        model.compile(loss='sparse_categorical_crossentropy',
                     optimizer='adam',
                     metrics=['accuracy'])
        
        self.model = model
        return model
    
    def prepare_data(self, text: str) -> tuple:
        """Prepare data for training.
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (input sequences, target sequences)
        """
        # Tokenize text
        self.tokenizer.fit_on_texts([text])
        total_words = len(self.tokenizer.word_index) + 1
        
        # Create input sequences
        input_sequences = []
        for line in text.split('\n'):
            token_list = self.tokenizer.texts_to_sequences([line])[0]
            for i in range(1, len(token_list)):
                n_gram_sequence = token_list[:i+1]
                input_sequences.append(n_gram_sequence)
        
        # Pad sequences
        max_sequence_len = max([len(x) for x in input_sequences])
        input_sequences = np.array(pad_sequences(input_sequences, 
                                               maxlen=max_sequence_len, 
                                               padding=self.padding))
        
        # Create predictors and labels
        predictors, labels = input_sequences[:,:-1], input_sequences[:,-1]
        labels = tf.keras.utils.to_categorical(labels, num_classes=total_words)
        
        return predictors, labels
    
    def train(self, 
             X_train: np.ndarray,
             y_train: np.ndarray,
             validation_data: Optional[tuple] = None,
             epochs: int = 100,
             batch_size: int = 128) -> tf.keras.callbacks.History:
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
    
    def generate_text(self, 
                     seed_text: str,
                     next_words: int = 100,
                     temperature: float = 1.0) -> str:
        """Generate text based on seed text.
        
        Args:
            seed_text: Starting text for generation
            next_words: Number of words to generate
            temperature: Sampling temperature (higher = more random)
            
        Returns:
            Generated text
        """
        if self.model is None:
            raise ValueError("Model not trained. Train the model first.")
            
        for _ in range(next_words):
            token_list = self.tokenizer.texts_to_sequences([seed_text])[0]
            token_list = pad_sequences([token_list], 
                                     maxlen=self.maxlen-1,
                                     padding=self.padding)
            predicted = self.model.predict(token_list, verbose=0)
            
            # Apply temperature
            predicted = np.log(predicted) / temperature
            exp_preds = np.exp(predicted)
            predicted = exp_preds / np.sum(exp_preds)
            
            # Sample from the distribution
            probas = np.random.multinomial(1, predicted[0], 1)
            predicted_index = np.argmax(probas)
            
            # Convert index to word
            output_word = ""
            for word, index in self.tokenizer.word_index.items():
                if index == predicted_index:
                    output_word = word
                    break
            
            seed_text += " " + output_word
            
        return seed_text 