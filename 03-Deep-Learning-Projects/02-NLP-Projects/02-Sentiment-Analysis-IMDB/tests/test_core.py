"""Tests for core functionality."""

import pytest
import numpy as np
from src.core.base import SentimentAnalyzer

def test_sentiment_analyzer_initialization():
    """Test SentimentAnalyzer initialization with default parameters."""
    analyzer = SentimentAnalyzer()
    assert analyzer.num_words == 1000
    assert analyzer.embedding_dim == 16
    assert analyzer.maxlen == 120
    assert analyzer.padding == 'post'
    assert analyzer.oov_token == "<OOV>"
    assert analyzer.model is None

def test_model_creation():
    """Test model creation with specific number of classes."""
    analyzer = SentimentAnalyzer()
    model = analyzer.create_model(num_classes=5)
    
    assert model is not None
    assert len(model.layers) == 4
    assert isinstance(model.layers[0], tf.keras.layers.Embedding)
    assert model.layers[-1].units == 5

def test_data_preparation():
    """Test data preparation with sample texts."""
    analyzer = SentimentAnalyzer(maxlen=10)
    texts = ["this is a test", "another test text"]
    labels = [0, 1]
    
    # Test without labels
    padded = analyzer.prepare_data(texts)
    assert isinstance(padded, np.ndarray)
    assert padded.shape[0] == 2
    assert padded.shape[1] == 10
    
    # Test with labels
    padded, label_array = analyzer.prepare_data(texts, labels)
    assert isinstance(padded, np.ndarray)
    assert isinstance(label_array, np.ndarray)
    assert len(label_array) == 2

def test_predict_without_training():
    """Test prediction without training raises error."""
    analyzer = SentimentAnalyzer()
    texts = ["test text"]
    
    with pytest.raises(ValueError):
        analyzer.predict(texts)

def test_train_without_model():
    """Test training without creating model raises error."""
    analyzer = SentimentAnalyzer()
    X = np.array([[1, 2, 3]])
    y = np.array([0])
    
    with pytest.raises(ValueError):
        analyzer.train(X, y) 