"""Tests for core functionality."""

import pytest
import numpy as np
import tensorflow as tf
from src.core.base import PoetryGenerator

def test_poetry_generator_initialization():
    """Test PoetryGenerator initialization with default parameters."""
    generator = PoetryGenerator()
    assert generator.vocab_size == 5000
    assert generator.embedding_dim == 100
    assert generator.rnn_units == 150
    assert generator.maxlen == 40
    assert generator.padding == 'pre'
    assert generator.oov_token == "<OOV>"
    assert generator.model is None

def test_model_creation():
    """Test model creation."""
    generator = PoetryGenerator()
    model = generator.create_model()
    
    assert model is not None
    assert len(model.layers) == 4
    assert isinstance(model.layers[0], tf.keras.layers.Embedding)
    assert isinstance(model.layers[1], tf.keras.layers.Bidirectional)
    assert isinstance(model.layers[2], tf.keras.layers.Dense)
    assert isinstance(model.layers[3], tf.keras.layers.Dense)
    assert model.layers[3].units == generator.vocab_size

def test_sequence_preparation():
    """Test sequence preparation with sample text."""
    generator = PoetryGenerator(vocab_size=100)
    corpus = ["this is a test", "another test line"]
    
    X, y = generator.prepare_sequences(corpus)
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert len(X) == len(y)
    assert y.shape[1] == generator.vocab_size  # One-hot encoded

def test_poetry_generation():
    """Test poetry generation with mock model."""
    generator = PoetryGenerator(vocab_size=100)
    generator.create_model()
    
    # Create a simple mock model that always predicts the same word
    def mock_predict(x, verbose=0):
        return np.array([[0.0] * 99 + [1.0]])
    
    generator.model.predict = mock_predict
    generator.tokenizer.word_index = {'test': 1}
    generator.tokenizer.index_word = {1: 'test'}
    
    generated = generator.generate_poetry("hello", next_words=3)
    assert len(generated.split()) == 4  # seed word + 3 generated words
    assert "\n" in generated  # Should contain line breaks

def test_invalid_generation():
    """Test poetry generation without trained model raises error."""
    generator = PoetryGenerator()
    
    with pytest.raises(ValueError):
        generator.generate_poetry("test")

def test_invalid_training():
    """Test training without model raises error."""
    generator = PoetryGenerator()
    X = np.array([[1, 2, 3]])
    y = np.array([0])
    
    with pytest.raises(ValueError):
        generator.train(X, y) 