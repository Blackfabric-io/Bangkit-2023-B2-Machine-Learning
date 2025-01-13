"""Core classes and functions for poetry generation."""

from typing import List, Optional, Dict, Any, Tuple
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

class PoetryGenerator:
    """Main class for poetry generation using LSTM."""
    
    def __init__(self, 
                 vocab_size: int = 5000,
                 embedding_dim: int = 100,
                 rnn_units: int = 150,
                 maxlen: int = 40,
                 padding: str = 'pre',
                 oov_token: str = "<OOV>"):
        """Initialize the poetry generator.
        
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
        
    def create_model(self) -> tf.keras.Model:
        """Create and compile the model.
        
        Returns:
            Compiled tensorflow model
        """
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(self.vocab_size, self.embedding_dim),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.rnn_units, return_sequences=True)),
            tf.keras.layers.Dense(self.rnn_units, activation='relu'),
            tf.keras.layers.Dense(self.vocab_size, activation='softmax')
        ])
        
        model.compile(loss='categorical_crossentropy',
                     optimizer='adam',
                     metrics=['accuracy'])
        
        self.model = model
        return model
    
    def prepare_sequences(self, corpus: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Create n-gram sequences from corpus.
        
        Args:
            corpus: List of text lines
            
        Returns:
            Tuple of (input sequences, one-hot encoded labels)
        """
        # Fit tokenizer
        self.tokenizer.fit_on_texts(corpus)
        
        # Create sequences
        input_sequences = []
        for line in corpus:
            token_list = self.tokenizer.texts_to_sequences([line])[0]
            for i in range(1, len(token_list)):
                n_gram_sequence = token_list[:i+1]
                input_sequences.append(n_gram_sequence)
        
        # Pad sequences
        max_sequence_len = max([len(x) for x in input_sequences])
        input_sequences = np.array(pad_sequences(input_sequences, 
                                               maxlen=max_sequence_len,
                                               padding=self.padding))
        
        # Split into features and labels
        features = input_sequences[:,:-1]
        labels = input_sequences[:,-1]
        
        # One-hot encode labels
        one_hot_labels = to_categorical(labels, num_classes=self.vocab_size)
        
        return features, one_hot_labels
    
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
    
    def generate_poetry(self, 
                       seed_text: str,
                       next_words: int = 50,
                       temperature: float = 1.0) -> str:
        """Generate poetry based on seed text.
        
        Args:
            seed_text: Starting text for generation
            next_words: Number of words to generate
            temperature: Sampling temperature (higher = more random)
            
        Returns:
            Generated poetry
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
            
            # Add line breaks for poetry-like formatting
            if len(seed_text.split()) % 8 == 0:  # Break every 8 words
                seed_text += "\n"
            
        return seed_text 