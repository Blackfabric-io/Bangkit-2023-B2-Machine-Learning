"""Tests for utility functions."""

import pytest
from src.utils.helpers import remove_stopwords, split_data

def test_remove_stopwords():
    """Test stopwords removal from text."""
    text = "This is a test sentence with stopwords"
    processed = remove_stopwords(text)
    
    # Check stopwords are removed
    assert "is" not in processed
    assert "a" not in processed
    assert "with" not in processed
    
    # Check content words remain
    assert "test" in processed
    assert "sentence" in processed
    assert "stopwords" in processed
    
    # Check case conversion
    assert processed.islower()

def test_split_data():
    """Test data splitting functionality."""
    sentences = ["text1", "text2", "text3", "text4", "text5"]
    labels = ["label1", "label2", "label3", "label4", "label5"]
    
    # Test with default split ratio
    train_sent, train_lab, val_sent, val_lab = split_data(sentences, labels)
    assert len(train_sent) == 4  # 80% of 5
    assert len(val_sent) == 1    # 20% of 5
    assert len(train_lab) == len(train_sent)
    assert len(val_lab) == len(val_sent)
    
    # Test with custom split ratio
    train_sent, train_lab, val_sent, val_lab = split_data(sentences, labels, train_split=0.6)
    assert len(train_sent) == 3  # 60% of 5
    assert len(val_sent) == 2    # 40% of 5
    
    # Test data integrity
    assert all(s in sentences for s in train_sent + val_sent)
    assert all(l in labels for l in train_lab + val_lab) 