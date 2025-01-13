"""Tests for core functionality."""

import pytest
import numpy as np
import tensorflow as tf
from src.core.base import TextGenerator

def test_text_generator_initialization():
    """Test TextGenerator initialization with default parameters."""
    generator = TextGenerator()
    assert generator.vocab_size == 1000
    assert generator.embedding_dim == 100
    assert generator.rnn_units == 256
    assert generator.maxlen == 40
    assert generator.padding == 'post'
    assert generator.oov_token == "<OOV>"
    assert generator.model is None

def test_model_creation():
    """Test model creation."""
    generator = TextGenerator()
    model = generator.create_model()
    
    assert model is not None
    assert len(model.layers) == 3
    assert isinstance(model.layers[0], tf.keras.layers.Embedding)
    assert isinstance(model.layers[1], tf.keras.layers.LSTM)
    assert isinstance(model.layers[2], tf.keras.layers.Dense)

def test_data_preparation():
    """Test data preparation with sample text."""
    generator = TextGenerator(vocab_size=100)
    text = "hello world\nthis is a test"
    
    X, y = generator.prepare_data(text)
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert len(X) == len(y)

def test_text_generation():
    """Test text generation with mock model."""
    generator = TextGenerator(vocab_size=100)
    generator.create_model()
    
    # Create a simple mock model that always predicts the same word
    def mock_predict(x, verbose=0):
        return np.array([[0.0] * 99 + [1.0]])
    
    generator.model.predict = mock_predict
    generator.tokenizer.word_index = {'test': 1}
    generator.tokenizer.index_word = {1: 'test'}
    
    generated = generator.generate_text("hello", next_words=3)
    assert len(generated.split()) == 4  # seed word + 3 generated words

def test_invalid_generation():
    """Test text generation without trained model raises error."""
    generator = TextGenerator()
    
    with pytest.raises(ValueError):
        generator.generate_text("test")

def test_invalid_training():
    """Test training without model raises error."""
    generator = TextGenerator()
    X = np.array([[1, 2, 3]])
    y = np.array([0])
    
    with pytest.raises(ValueError):
        generator.train(X, y) 